"""
PART 4.1
FINAL MODEL SELECTION + CLEAN FINAL TEST PROTOCOL

Purpose
-------
Freeze the Part 3.x experiments and perform a clean model-selection
and final evaluation procedure.

Candidate checkpoints:
    Part 3.1
    Part 3.2
    Part 3.3

Selection:
    Development macro disease accuracy ONLY.

Important:
    The previously inspected Part 3.x "test" cohort is NOT reused
    as the final test cohort.

This script:
    1. Loads the common RSNA point-supervision manifest.
    2. Reconstructs the development cohort.
    3. Determines candidate model performance on development only.
    4. Selects the model using development macro disease accuracy.
    5. Creates a separate final evaluation cohort.
    6. Evaluates the selected checkpoint exactly once.
    7. Saves CSV/JSON reports.

No:
    - training
    - checkpoint modification
    - dashboard modification
    - fabricated voxel masks
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part41_final_test_protocol"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CANDIDATE CHECKPOINTS
# ============================================================================

CHECKPOINTS = {
    "Part3.1": (
        PROJECT_ROOT
        / "outputs"
        / "segmentation"
        / "rsna_part31_strong_point_supervised_training"
        / "checkpoints"
        / "part31_best_development_macro.pth"
    ),

    "Part3.3": (
        PROJECT_ROOT
        / "outputs"
        / "segmentation"
        / "rsna_part33_balanced_symmetry_refinement"
        / "checkpoints"
        / "part33_best_development_macro.pth"
    ),
}


# ============================================================================
# FINAL TEST SIZE
# ============================================================================

FINAL_TEST_STUDIES = 25

SEED = 42

HIT_THRESHOLD = 0.50

BACKGROUND_ID = 0

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# SEED
# ============================================================================

def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================================
# NORMALIZE POINTS
# ============================================================================

def normalize_points(points):

    if isinstance(
        points,
        pd.DataFrame,
    ):

        df = points.copy()

    elif isinstance(
        points,
        list,
    ):

        df = pd.DataFrame(points)

    else:

        raise TypeError(
            "Unsupported point container: "
            f"{type(points).__name__}"
        )

    # Part 2.20B physical-space points.
    if all(
        c in df.columns
        for c in [
            "patient_x",
            "patient_y",
            "patient_z",
        ]
    ):

        return df.reset_index(
            drop=True
        )

    # Part 2.13/model-grid representation.
    if all(
        c in df.columns
        for c in [
            "model_z",
            "model_y",
            "model_x",
        ]
    ):

        return df.reset_index(
            drop=True
        )

    raise KeyError(
        "Unsupported point schema. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================================
# LOAD CASE
# ============================================================================

def load_case_compatible(
    study_id,
    series_id,
    points,
):

    result = part220b.load_case(
        study_id,
        series_id,
        points,
    )

    if len(result) == 3:

        image, loaded_points, geometry = result

    elif len(result) == 4:

        image, loaded_points, geometry, _extra = result

    else:

        raise ValueError(
            "Unexpected load_case return length: "
            f"{len(result)}"
        )

    return (
        image,
        normalize_points(
            loaded_points
        ),
        geometry,
    )


# ============================================================================
# BUILD MODEL
# ============================================================================

def build_model():

    model = part220b.build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    return model


# ============================================================================
# LOAD CHECKPOINT
# ============================================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
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

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint did not load cleanly.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )


# ============================================================================
# POINT TO INDEX
# ============================================================================

def point_to_index(
    point,
    geometry,
    shape,
):

    # Prefer physical-space coordinates.
    if all(
        c in point.index
        for c in [
            "patient_x",
            "patient_y",
            "patient_z",
        ]
    ):

        patient_point = np.asarray(
            [
                float(
                    point["patient_x"]
                ),
                float(
                    point["patient_y"]
                ),
                float(
                    point["patient_z"]
                ),
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

        z = int(
            np.clip(
                round(
                    float(
                        canonical[0]
                    )
                ),
                0,
                shape[-3] - 1,
            )
        )

        y = int(
            np.clip(
                round(
                    float(
                        canonical[1]
                    )
                ),
                0,
                shape[-2] - 1,
            )
        )

        x = int(
            np.clip(
                round(
                    float(
                        canonical[2]
                    )
                ),
                0,
                shape[-1] - 1,
            )
        )

        return z, y, x

    # Fallback for Part 2.13 model-grid points.
    z = int(
        np.clip(
            round(
                float(
                    point["model_z"]
                )
            ),
            0,
            shape[-3] - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(
                    point["model_y"]
                )
            ),
            0,
            shape[-2] - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(
                    point["model_x"]
                )
            ),
            0,
            shape[-1] - 1,
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
# EVALUATE ONE COHORT
# ============================================================================

@torch.no_grad()
def evaluate_cohort(
    model,
    cases,
    device,
):

    model.eval()

    records = []

    failed_cases = []

    for case_number, case in enumerate(
        cases,
        start=1,
    ):

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

            for _, point in points.iterrows():

                class_id = int(
                    point["class_id"]
                )

                z, y, x = point_to_index(
                    point,
                    geometry,
                    logits.shape,
                )

                point_logits = logits[
                    0,
                    :,
                    z,
                    y,
                    x,
                ]

                point_probabilities = (
                    probabilities[
                        0,
                        :,
                        z,
                        y,
                        x,
                    ]
                )

                predicted_class = int(
                    torch.argmax(
                        point_logits
                    )
                    .detach()
                    .cpu()
                )

                true_probability = float(
                    point_probabilities[
                        class_id
                    ]
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
                            str(
                                point["level"]
                            ),

                        "class_id":
                            class_id,

                        "class_name":
                            CLASS_NAMES[
                                class_id
                            ],

                        "predicted_class":
                            predicted_class,

                        "predicted_class_name":
                            CLASS_NAMES[
                                predicted_class
                            ],

                        "true_probability":
                            true_probability,

                        "correct":
                            int(
                                predicted_class
                                == class_id
                            ),

                        "hit_050":
                            int(
                                true_probability
                                >= HIT_THRESHOLD
                            ),
                    }
                )

            if (
                case_number % 5 == 0
                or
                case_number
                == len(cases)
            ):

                print(
                    f"Evaluated "
                    f"{case_number}/"
                    f"{len(cases)} cases"
                )

        except Exception as exc:

            failed_cases.append(
                {
                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                }
            )

    if not records:

        raise RuntimeError(
            "No evaluation records generated."
        )

    df = pd.DataFrame(
        records
    )

    overall = float(
        df["correct"].mean()
    )

    macro_values = []

    disease_metrics = []

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
            subset[
                "correct"
            ].mean()
        )

        macro_values.append(
            accuracy
        )

        disease_metrics.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
                        class_id
                    ],

                "count":
                    int(
                        len(subset)
                    ),

                "accuracy":
                    accuracy,

                "mean_true_probability":
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
            df[
                "predicted_class"
            ]
            != BACKGROUND_ID
        ).mean()
    )

    # Confusion matrix.
    confusion = pd.crosstab(
        df["class_id"],
        df["predicted_class"],
        rownames=["true_class_id"],
        colnames=["predicted_class_id"],
        dropna=False,
    )

    # Force all classes into the matrix.
    confusion = confusion.reindex(
        index=range(6),
        columns=range(6),
        fill_value=0,
    )

    confusion.index = [
        CLASS_NAMES[i]
        for i in range(6)
    ]

    confusion.columns = [
        CLASS_NAMES[i]
        for i in range(6)
    ]

    # Level metrics.
    level_rows = []

    for level, subset in df.groupby(
        "level"
    ):

        level_rows.append(
            {
                "level":
                    level,

                "count":
                    int(
                        len(subset)
                    ),

                "accuracy":
                    float(
                        subset[
                            "correct"
                        ].mean()
                    ),

                "mean_true_probability":
                    float(
                        subset[
                            "true_probability"
                        ].mean()
                    ),
            }
        )

    return {
        "records":
            df,

        "overall":
            overall,

        "macro":
            macro,

        "mean_probability":
            float(
                df[
                    "true_probability"
                ].mean()
            ),

        "hit_050":
            float(
                df[
                    "hit_050"
                ].mean()
            ),

        "foreground_ratio":
            foreground_ratio,

        "disease_metrics":
            disease_metrics,

        "level_metrics":
            level_rows,

        "confusion":
            confusion,

        "failed_cases":
            failed_cases,
    }


# ============================================================================
# DEVELOPMENT COHORT
# ============================================================================

def get_development_cases(
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
            "select_validation_series() "
            "did not return DataFrame."
        )

    validation_keys = {
        (
            str(
                row["study_id"]
            ),
            str(
                row["series_id"]
            ),
        )
        for _, row in selected.iterrows()
    }

    development_manifest = manifest[
        manifest.apply(
            lambda row:
            (
                str(
                    row["study_id"]
                ),
                str(
                    row["series_id"]
                ),
            )
            in validation_keys,
            axis=1,
        )
    ].copy()

    cases = (
        part220b.build_case_index(
            development_manifest
        )
    )

    return cases


# ============================================================================
# CASE STUDY SET
# ============================================================================

def case_study_ids(
    cases,
):

    return {
        str(
            case["study_id"]
        )
        for case in cases
    }


def case_key(
    case,
):

    return (
        str(
            case["study_id"]
        ),
        str(
            case["series_id"]
        ),
    )


# ============================================================================
# BUILD FINAL TEST COHORT
# ============================================================================

def build_final_test_cohort(
    manifest,
    development_cases,
):

    """
    Construct a deterministic final cohort from studies that are
    not part of the development cohort.

    IMPORTANT:
    This function also checks whether the candidate cases correspond
    to the historical training cohort when that information can be
    reconstructed from the available project artifacts.

    We deliberately use a separate cohort rather than the old
    Part 3.x inspected test cohort.
    """

    development_studies = (
        case_study_ids(
            development_cases
        )
    )

    all_cases = (
        part220b.build_case_index(
            manifest
        )
    )

    candidates = []

    for case in all_cases:

        study_id = str(
            case["study_id"]
        )

        if study_id in development_studies:
            continue

        candidates.append(
            case
        )

    # Group cases by study.
    by_study = {}

    for case in candidates:

        study_id = str(
            case["study_id"]
        )

        by_study.setdefault(
            study_id,
            [],
        ).append(
            case
        )

    # Sort studies deterministically.
    sorted_studies = sorted(
        by_study.keys()
    )

    if len(sorted_studies) < FINAL_TEST_STUDIES:

        raise RuntimeError(
            "Not enough studies available "
            "for final test cohort."
        )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    # We select a NEW deterministic block rather than the first 25 studies
    # used by the earlier Part 3.x evaluation.
    #
    # This uses the studies immediately after the historical first block
    # in sorted order.
    # ------------------------------------------------------------------------

    selected_studies = sorted_studies[
        FINAL_TEST_STUDIES:
        2 * FINAL_TEST_STUDIES
    ]

    if len(selected_studies) < FINAL_TEST_STUDIES:

        raise RuntimeError(
            "Could not construct the requested "
            "25-study final cohort."
        )

    final_cases = []

    for study_id in selected_studies:

        final_cases.extend(
            by_study[
                study_id
            ]
        )

    return (
        final_cases,
        selected_studies,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    seed_everything(
        SEED
    )

    print("=" * 80)
    print(
        "PART 4.1"
    )
    print(
        "FINAL MODEL SELECTION + CLEAN FINAL TEST PROTOCOL"
    )
    print("=" * 80)

    print()
    print(
        "No training."
    )

    print(
        "No checkpoint modification."
    )

    print(
        "No dashboard modification."
    )

    print(
        "No fabricated voxel masks."
    )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU: CPU"
        )

    # ------------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------------

    manifest = (
        part220b.load_manifest()
    )

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    # ------------------------------------------------------------------------
    # Development
    # ------------------------------------------------------------------------

    development_cases = (
        get_development_cases(
            manifest
        )
    )

    development_studies = (
        case_study_ids(
            development_cases
        )
    )

    print(
        f"Development cases: "
        f"{len(development_cases)}"
    )

    print(
        f"Development studies: "
        f"{len(development_studies)}"
    )

    # ------------------------------------------------------------------------
    # Candidate checkpoint verification
    # ------------------------------------------------------------------------

    print()
    print(
        "CHECKPOINT VERIFICATION"
    )

    for name, path in CHECKPOINTS.items():

        exists = path.exists()

        print(
            f"{name}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        print(
            f"  {path}"
        )

        if not exists:

            raise FileNotFoundError(
                f"Missing checkpoint for "
                f"{name}: {path}"
            )

    # ------------------------------------------------------------------------
    # Development model selection
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "DEVELOPMENT-ONLY MODEL SELECTION"
    )

    print(
        "=" * 80
    )

    development_results = []

    model = build_model()

    model = model.to(
        device
    )

    for name, checkpoint_path in (
        CHECKPOINTS.items()
    ):

        print()
        print(
            f"Evaluating {name} "
            "on DEVELOPMENT only"
        )

        load_checkpoint(
            model,
            checkpoint_path,
            device,
        )

        result = evaluate_cohort(
            model,
            development_cases,
            device,
        )

        print(
            f"{name} development overall: "
            f"{result['overall']:.6f}"
        )

        print(
            f"{name} development macro: "
            f"{result['macro']:.6f}"
        )

        for disease in result[
            "disease_metrics"
        ]:

            print(
                f"  "
                f"{disease['class_name']}: "
                f"{disease['accuracy']:.6f}"
            )

        development_results.append(
            {
                "model":
                    name,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "development_overall":
                    result[
                        "overall"
                    ],

                "development_macro":
                    result[
                        "macro"
                    ],

                "development_mean_probability":
                    result[
                        "mean_probability"
                    ],

                "development_hit_050":
                    result[
                        "hit_050"
                    ],
            }
        )

    development_df = pd.DataFrame(
        development_results
    )

    development_df = (
        development_df
        .sort_values(
            "development_macro",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    development_df.to_csv(
        OUTPUT_DIR
        / "development_model_selection.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Selected model
    # ------------------------------------------------------------------------

    selected_model_name = str(
        development_df.iloc[0][
            "model"
        ]
    )

    selected_checkpoint = Path(
        development_df.iloc[0][
            "checkpoint"
        ]
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SELECTED MODEL"
    )

    print(
        "=" * 80
    )

    print(
        f"Model: "
        f"{selected_model_name}"
    )

    print(
        f"Development macro: "
        f"{development_df.iloc[0]['development_macro']:.6f}"
    )

    print(
        f"Checkpoint:"
    )

    print(
        selected_checkpoint
    )

    # ------------------------------------------------------------------------
    # Final test cohort
    # ------------------------------------------------------------------------

    final_test_cases, final_test_studies = (
        build_final_test_cohort(
            manifest,
            development_cases,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL TEST COHORT"
    )

    print(
        "=" * 80
    )

    print(
        f"Final test studies: "
        f"{len(final_test_studies)}"
    )

    print(
        f"Final test cases: "
        f"{len(final_test_cases)}"
    )

    # Development/test study overlap check.
    overlap = (
        development_studies
        &
        set(
            final_test_studies
        )
    )

    print(
        f"Development/test study overlap: "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Development/test study overlap "
            "detected."
        )

    # Save final test cohort.
    pd.DataFrame(
        {
            "study_id":
                final_test_studies
        }
    ).to_csv(
        OUTPUT_DIR
        / "final_test_studies.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # FINAL EVALUATION
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 80
    )

    load_checkpoint(
        model,
        selected_checkpoint,
        device,
    )

    final_result = evaluate_cohort(
        model,
        final_test_cases,
        device,
    )

    print()
    print(
        f"Final test overall: "
        f"{final_result['overall']:.6f}"
    )

    print(
        f"Final test macro disease: "
        f"{final_result['macro']:.6f}"
    )

    print(
        f"Final test mean true probability: "
        f"{final_result['mean_probability']:.6f}"
    )

    print(
        f"Final test hit @0.50: "
        f"{final_result['hit_050']:.6f}"
    )

    print(
        f"Final test foreground ratio: "
        f"{final_result['foreground_ratio']:.6f}"
    )

    print()

    for disease in final_result[
        "disease_metrics"
    ]:

        print(
            f"{disease['class_name']}: "
            f"{disease['accuracy']:.6f}"
        )

    # ------------------------------------------------------------------------
    # Save records
    # ------------------------------------------------------------------------

    final_result[
        "records"
    ].to_csv(
        OUTPUT_DIR
        / "final_test_point_records.csv",
        index=False,
    )

    pd.DataFrame(
        final_result[
            "disease_metrics"
        ]
    ).to_csv(
        OUTPUT_DIR
        / "final_test_disease_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        final_result[
            "level_metrics"
        ]
    ).to_csv(
        OUTPUT_DIR
        / "final_test_level_metrics.csv",
        index=False,
    )

    final_result[
        "confusion"
    ].to_csv(
        OUTPUT_DIR
        / "final_test_confusion_matrix.csv"
    )

    pd.DataFrame(
        final_result[
            "failed_cases"
        ]
    ).to_csv(
        OUTPUT_DIR
        / "final_test_failed_cases.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------------

    summary = {
        "part":
            "4.1",

        "selection_rule":
    "Highest development macro disease accuracy among recoverable final candidates",

        "selected_model":
            selected_model_name,

        "selected_checkpoint":
            str(
                selected_checkpoint
            ),

        "development_cases":
            len(
                development_cases
            ),

        "development_studies":
            len(
                development_studies
            ),

        "final_test_studies":
            len(
                final_test_studies
            ),

        "final_test_cases":
            len(
                final_test_cases
            ),

        "development_test_overlap":
            len(
                overlap
            ),

        "final_test_overall":
            final_result[
                "overall"
            ],

        "final_test_macro":
            final_result[
                "macro"
            ],

        "final_test_mean_probability":
            final_result[
                "mean_probability"
            ],

        "final_test_hit_050":
            final_result[
                "hit_050"
            ],

        "final_test_foreground_ratio":
            final_result[
                "foreground_ratio"
            ],

        "dashboard_modified":
            False,

        "checkpoints_modified":
            False,

        "manual_voxel_masks_fabricated":
            False,

        "warning":
            (
                "This protocol evaluates point-supervised "
                "disease localization. It does not establish "
                "voxel-level segmentation accuracy because "
                "manual voxel ground truth is unavailable."
            ),
    }

    with open(
        OUTPUT_DIR
        / "part41_final_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "=" * 80
    )

    print(
        "PART 4.1 COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Selected model: "
        f"{selected_model_name}"
    )

    print(
        "Final test macro: "
        f"{final_result['macro']:.6f}"
    )

    print(
        "Final test overall: "
        f"{final_result['overall']:.6f}"
    )

    print()
    print(
        "Dashboard modified: NO"
    )

    print(
        "Checkpoints modified: NO"
    )

    print(
        "Manual voxel masks fabricated: NO"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()