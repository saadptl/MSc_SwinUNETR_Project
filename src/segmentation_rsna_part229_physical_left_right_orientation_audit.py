"""
PART 2.29 — PHYSICAL LEFT/RIGHT FORAMINAL ORIENTATION AUDIT

Purpose
-------
Investigate the persistent LFNN/RFNN asymmetry observed in Part 2.27
and Part 2.28.

This is an ANALYSIS-ONLY experiment.

It:
    - uses the exact Part 2.20B geometry pipeline
    - uses the Part 2.27 best checkpoint
    - uses the same 25 validation cases
    - analyzes LFNN/RFNN annotations
    - compares native X, patient physical X and canonical X
    - identifies paired LFNN/RFNN series
    - examines prediction behavior

It does NOT:
    - train
    - modify checkpoints
    - modify Part 2.20B
    - modify Part104
    - modify the dashboard
    - fabricate voxel ground truth
"""

from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXACT PART 2.20B PIPELINE
# ============================================================

from src import segmentation_rsna_part220b_geometry_corrected_training as p220b


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part229_physical_left_right_orientation_audit"
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


# ============================================================
# CLASSES
# ============================================================

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

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

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

    if missing:
        print("Missing keys:")
        for key in missing[:10]:
            print("  ", key)

    if unexpected:
        print("Unexpected keys:")
        for key in unexpected[:10]:
            print("  ", key)


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
    )

    probabilities = (
        probabilities[0]
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
# VALIDATION GROUPS
# ============================================================

def build_validation_groups(
    manifest,
):

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
                            item[
                                "study_id"
                            ]
                        ),
                        str(
                            item[
                                "series_id"
                            ]
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PART 2.29 — PHYSICAL LEFT/RIGHT "
        "FORAMINAL ORIENTATION AUDIT"
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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

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

    point_records = []

    processed = 0

    # --------------------------------------------------------
    # Process each series
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

        # ----------------------------------------------------
        # Native coordinates must come from ORIGINAL manifest
        # ----------------------------------------------------

        native_lookup = {}

        for _, row in point_df.iterrows():

            key = (
                int(row["class_id"]),
                str(row["level"]),
            )

            native_lookup[key] = {
                "native_z": safe_float(
                    row["native_z"]
                ),
                "native_y": safe_float(
                    row["native_y"]
                ),
                "native_x": safe_float(
                    row["native_x"]
                ),
            }

        # ----------------------------------------------------
        # Analyze only LFNN/RFNN
        # ----------------------------------------------------

        for point in points:

            class_id = int(
                point["class_id"]
            )

            if class_id not in {
                LFNN,
                RFNN,
            }:
                continue

            level = str(
                point["level"]
            )

            lookup_key = (
                class_id,
                level,
            )

            native = native_lookup.get(
                lookup_key,
                {},
            )

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

            point_probs = probabilities[
                :,
                z,
                y,
                x,
            ]

            predicted_class = int(
                np.argmax(
                    point_probs
                )
            )

            # ------------------------------------------------
            # Local foreground
            # ------------------------------------------------

            def local_fg(radius):

                z0 = max(
                    0,
                    z - radius,
                )

                z1 = min(
                    probabilities.shape[1],
                    z + radius + 1,
                )

                y0 = max(
                    0,
                    y - radius,
                )

                y1 = min(
                    probabilities.shape[2],
                    y + radius + 1,
                )

                x0 = max(
                    0,
                    x - radius,
                )

                x1 = min(
                    probabilities.shape[3],
                    x + radius + 1,
                )

                local_pred = prediction[
                    z0:z1,
                    y0:y1,
                    x0:x1,
                ]

                return float(
                    np.mean(
                        local_pred != 0
                    )
                )

            point_records.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "level": level,
                    "true_class": class_id,
                    "true_class_name": CLASS_NAMES[
                        class_id
                    ],
                    "native_z": native.get(
                        "native_z",
                        np.nan,
                    ),
                    "native_y": native.get(
                        "native_y",
                        np.nan,
                    ),
                    "native_x": native.get(
                        "native_x",
                        np.nan,
                    ),
                    "patient_x": safe_float(
                        point[
                            "patient_x"
                        ]
                    ),
                    "patient_y": safe_float(
                        point[
                            "patient_y"
                        ]
                    ),
                    "patient_z": safe_float(
                        point[
                            "patient_z"
                        ]
                    ),
                    "canonical_z": safe_float(
                        point["z"]
                    ),
                    "canonical_y": safe_float(
                        point["y"]
                    ),
                    "canonical_x": safe_float(
                        point["x"]
                    ),
                    "predicted_class": predicted_class,
                    "predicted_class_name":
                        CLASS_NAMES[
                            predicted_class
                        ],
                    "true_probability":
                        safe_float(
                            point_probs[
                                class_id
                            ]
                        ),
                    "predicted_probability":
                        safe_float(
                            point_probs[
                                predicted_class
                            ]
                        ),
                    "foreground_radius2":
                        local_fg(2),
                    "foreground_radius4":
                        local_fg(4),
                    "correct": int(
                        predicted_class
                        == class_id
                    ),
                }
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
        point_records
    )

    if df.empty:
        raise RuntimeError(
            "No LFNN/RFNN points were generated."
        )

    point_path = (
        TABLE_DIR
        / "part229_foraminal_points.csv"
    )

    df.to_csv(
        point_path,
        index=False,
    )

    # ========================================================
    # PAIR LFNN/RFNN WITHIN SAME SERIES + LEVEL
    # ========================================================

    pair_rows = []

    grouped = df.groupby(
        [
            "study_id",
            "series_id",
            "level",
        ],
        sort=True,
    )

    for (
        study_id,
        series_id,
        level,
    ), group in grouped:

        left = group[
            group["true_class"]
            == LFNN
        ]

        right = group[
            group["true_class"]
            == RFNN
        ]

        if left.empty or right.empty:
            continue

        # There should normally be one point per
        # disease/level/series combination.

        left_row = left.iloc[0]
        right_row = right.iloc[0]

        pair_rows.append(
            {
                "study_id": study_id,
                "series_id": series_id,
                "level": level,

                "lfnn_patient_x":
                    safe_float(
                        left_row[
                            "patient_x"
                        ]
                    ),

                "rfnn_patient_x":
                    safe_float(
                        right_row[
                            "patient_x"
                        ]
                    ),

                "lfnn_native_x":
                    safe_float(
                        left_row[
                            "native_x"
                        ]
                    ),

                "rfnn_native_x":
                    safe_float(
                        right_row[
                            "native_x"
                        ]
                    ),

                "lfnn_canonical_x":
                    safe_float(
                        left_row[
                            "canonical_x"
                        ]
                    ),

                "rfnn_canonical_x":
                    safe_float(
                        right_row[
                            "canonical_x"
                        ]
                    ),

                "lfnn_correct":
                    int(
                        left_row[
                            "correct"
                        ]
                    ),

                "rfnn_correct":
                    int(
                        right_row[
                            "correct"
                        ]
                    ),

                "rfnn_predicted_class":
                    right_row[
                        "predicted_class_name"
                    ],

                "rfnn_true_probability":
                    safe_float(
                        right_row[
                            "true_probability"
                        ]
                    ),

                "rfnn_foreground_radius2":
                    safe_float(
                        right_row[
                            "foreground_radius2"
                        ]
                    ),
            }
        )

    pairs = pd.DataFrame(
        pair_rows
    )

    pairs_path = (
        TABLE_DIR
        / "part229_lfnn_rfnn_pairs.csv"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    # ========================================================
    # ORIENTATION CONSISTENCY
    # ========================================================

    def ordering_accuracy(
        data,
        left_column,
        right_column,
    ):

        valid = data[
            data[
                left_column
            ].notna()
            & data[
                right_column
            ].notna()
        ].copy()

        if valid.empty:
            return {
                "pairs": 0,
                "left_less_right": np.nan,
                "left_greater_right": np.nan,
                "equal": 0,
            }

        left_less = (
            valid[
                left_column
            ]
            < valid[
                right_column
            ]
        )

        left_greater = (
            valid[
                left_column
            ]
            > valid[
                right_column
            ]
        )

        equal = (
            valid[
                left_column
            ]
            == valid[
                right_column
            ]
        )

        return {
            "pairs": int(
                len(valid)
            ),
            "left_less_right": float(
                left_less.mean()
            ),
            "left_greater_right": float(
                left_greater.mean()
            ),
            "equal": int(
                equal.sum()
            ),
        }

    orientation = {
        "patient_x": ordering_accuracy(
            pairs,
            "lfnn_patient_x",
            "rfnn_patient_x",
        ),
        "native_x": ordering_accuracy(
            pairs,
            "lfnn_native_x",
            "rfnn_native_x",
        ),
        "canonical_x": ordering_accuracy(
            pairs,
            "lfnn_canonical_x",
            "rfnn_canonical_x",
        ),
    }

    with open(
        REPORT_DIR
        / "part229_orientation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            orientation,
            f,
            indent=2,
        )

    # ========================================================
    # RFNN BACKGROUND ANALYSIS
    # ========================================================

    rfnn = df[
        df["true_class"]
        == RFNN
    ].copy()

    rfnn_background = (
        rfnn[
            rfnn[
                "predicted_class"
            ]
            == 0
        ]
    )

    rfnn_non_background = (
        rfnn[
            rfnn[
                "predicted_class"
            ]
            != 0
        ]
    )

    # ========================================================
    # REPORT
    # ========================================================

    lines = []

    lines.append(
        "PART 2.29 — PHYSICAL LEFT/RIGHT "
        "FORAMINAL ORIENTATION AUDIT"
    )

    lines.append("")

    lines.append(
        f"Validation cases processed: "
        f"{processed}/{len(validation_groups)}"
    )

    lines.append(
        f"LFNN/RFNN points analyzed: "
        f"{len(df)}"
    )

    lines.append(
        f"Paired LFNN/RFNN observations: "
        f"{len(pairs)}"
    )

    lines.append("")

    lines.append(
        "ORIENTATION ORDERING"
    )

    for name, values in orientation.items():

        lines.append(
            f"{name}: "
            f"pairs={values['pairs']}, "
            f"LFNN<X<R FNN="
            f"{values['left_less_right']}, "
            f"LFNN>RFNN="
            f"{values['left_greater_right']}, "
            f"equal={values['equal']}"
        )

    lines.append("")

    lines.append(
        "RFNN PREDICTION SUMMARY"
    )

    lines.append(
        f"RFNN points: {len(rfnn)}"
    )

    lines.append(
        f"RFNN background predictions: "
        f"{len(rfnn_background)}"
    )

    lines.append(
        f"RFNN non-background predictions: "
        f"{len(rfnn_non_background)}"
    )

    if len(rfnn):

        lines.append(
            "RFNN background fraction: "
            f"{len(rfnn_background) / len(rfnn):.6f}"
        )

        lines.append(
            "RFNN mean patient X: "
            f"{rfnn['patient_x'].mean():.6f}"
        )

        lines.append(
            "RFNN mean canonical X: "
            f"{rfnn['canonical_x'].mean():.6f}"
        )

        lines.append(
            "RFNN mean radius-2 foreground: "
            f"{rfnn['foreground_radius2'].mean():.6f}"
        )

    lines.append("")

    lines.append(
        "LFNN SUMMARY"
    )

    lfnn = df[
        df["true_class"]
        == LFNN
    ]

    lines.append(
        f"LFNN accuracy: "
        f"{lfnn['correct'].mean():.6f}"
    )

    lines.append(
        f"LFNN mean radius-2 foreground: "
        f"{lfnn['foreground_radius2'].mean():.6f}"
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
        "RSNA annotations remain "
        "point/localization annotations."
    )

    lines.append(
        "This audit does not establish "
        "voxel-wise segmentation accuracy."
    )

    report_path = (
        REPORT_DIR
        / "part229_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL CONSOLE
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PART 2.29 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Processed cases: "
        f"{processed}/{len(validation_groups)}"
    )

    print(
        f"LFNN/RFNN points: "
        f"{len(df)}"
    )

    print(
        f"Paired LFNN/RFNN observations: "
        f"{len(pairs)}"
    )

    print()

    for name, values in orientation.items():

        print(
            f"{name}: "
            f"pairs={values['pairs']} | "
            f"LFNN<X<R FNN="
            f"{values['left_less_right']:.6f} | "
            f"LFNN>RFNN="
            f"{values['left_greater_right']:.6f}"
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
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()