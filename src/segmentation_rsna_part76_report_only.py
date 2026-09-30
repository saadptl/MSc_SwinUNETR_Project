"""
PART 76 — REPORT-ONLY

Purpose
-------
Generate the complete Part 76 report from the EXISTING Part 76 checkpoints.

IMPORTANT:
- NO TRAINING
- NO BACKWARD PASS
- NO OPTIMIZER STEP
- NO CHECKPOINT MODIFICATION
- Uses the existing Part 76 E1/E2/E3 checkpoints.
- Re-evaluates the 50-case validation cohort only.

Conditions:
A = R2_FULL_CLASS_BALANCED
B = R2_FULL_CLASS_BALANCED_POINT_AUX

This script reconstructs the missing reporting artifacts after the original
Part 76 experiment completed but failed during final DataFrame assembly.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PATHS
# ============================================================================
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

# The existing Part 76 implementation supplies the validated loading,
# coordinate mapping, model creation, class-balanced loss, and evaluation code.
CANDIDATES = [
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision.py",
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision_FIXED_v2.py",
    SRC / "segmentation_rsna_part76_c2_c3_point_auxiliary_supervision_REPORTING_FIXED.py",
]

PART76_SOURCE = next(
    (p for p in CANDIDATES if p.exists()),
    None,
)

if PART76_SOURCE is None:
    raise FileNotFoundError(
        "Could not find an existing Part 76 source script in src."
    )

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

TRAIN_CSV = P15 / "part15_train_cohort.csv"
VAL_CSV = P15 / "part15_validation_cohort.csv"
INIT_CKPT = (
    P15
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

PART76_OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part76_c2_c3_point_auxiliary_supervision"
)

CKPT = PART76_OUT / "checkpoints"
REPORT = PART76_OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================
TRAIN_N = 100
VAL_N = 50
EPOCHS = 3
SEED = 42

CONDITIONS = [
    "R2_FULL_CLASS_BALANCED",
    "R2_FULL_CLASS_BALANCED_POINT_AUX",
]


# ============================================================================
# HELPERS
# ============================================================================
def banner(text: str) -> None:
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def checkpoint_path(condition: str, epoch: int) -> Path:
    return (
        CKPT
        / f"part76_{condition.lower()}_epoch{epoch}.pth"
    )


def validate_paths() -> None:
    banner("PART 76 REPORT-ONLY PATH VALIDATION")

    required = {
        "Part 76 source": PART76_SOURCE,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
    }

    missing = []

    for name, path in required.items():
        ok = path.exists()
        print(
            f"{name:<34}: "
            f"{'FOUND' if ok else 'MISSING'}"
        )
        if not ok:
            missing.append(str(path))

    for condition in CONDITIONS:
        for epoch in range(1, EPOCHS + 1):
            p = checkpoint_path(condition, epoch)
            ok = p.exists()
            print(
                f"{condition} E{epoch:<24}: "
                f"{'FOUND' if ok else 'MISSING'}"
            )
            if not ok:
                missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 76 artifacts:\n"
            + "\n".join(missing)
        )


# ============================================================================
# BEST-EPOCH LOCALIZATION SUMMARY
# ============================================================================
def make_c2_c3_localization(
    class_df: pd.DataFrame,
    point_df: pd.DataFrame,
    condition: str,
    epoch: int,
):
    rows = []

    for c in (2, 3):
        g = class_df[
            class_df["class_id"] == c
        ].copy()

        p = point_df[
            point_df["class_id"] == c
        ].copy()

        target = int(g["target_voxels"].sum())
        pred = int(g["predicted_voxels"].sum())
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())

        def finite_mean(column):
            if len(p) == 0:
                return np.nan
            x = pd.to_numeric(
                p[column],
                errors="coerce",
            )
            x = x.replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()
            return (
                float(x.mean())
                if len(x)
                else np.nan
            )

        rows.append(
            {
                "condition": condition,
                "epoch": epoch,
                "class_id": c,
                "class_name": class_df[
                    class_df["class_id"] == c
                ]["class_name"].iloc[0],
                "target_voxels": target,
                "predicted_voxels": pred,
                "prediction_target_ratio": safe_div(
                    pred,
                    target,
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": safe_div(
                    2 * tp,
                    pred + target,
                ),
                "precision": safe_div(
                    tp,
                    tp + fp,
                ),
                "recall": safe_div(
                    tp,
                    tp + fn,
                ),
                "point_count": len(p),
                "point_on_target_fraction": (
                    float(
                        p["point_on_target"].mean()
                    )
                    if len(p)
                    else np.nan
                ),
                "point_on_prediction_fraction": (
                    float(
                        p["point_on_prediction"].mean()
                    )
                    if len(p)
                    else np.nan
                ),
                "annotation_to_target_mean_distance": (
                    finite_mean(
                        "annotation_to_target_distance"
                    )
                ),
                "annotation_to_prediction_mean_distance": (
                    finite_mean(
                        "annotation_to_prediction_distance"
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================
def main():
    banner("PART 76 REPORT-ONLY")
    print(
        "This script ONLY evaluates existing Part 76 checkpoints "
        "and regenerates the reports."
    )
    print("")
    print("NO TRAINING")
    print("NO BACKWARD PASS")
    print("NO OPTIMIZER STEP")
    print("NO CHECKPOINT MODIFICATION")

    validate_paths()

    p76 = load_module(
        PART76_SOURCE,
        "part76_report_only_source",
    )

    device = p76.DEVICE

    train_rows = pd.read_csv(
        TRAIN_CSV
    ).head(TRAIN_N).copy()

    val_rows = pd.read_csv(
        VAL_CSV
    ).head(VAL_N).copy()

    coord = pd.read_csv(
        p76.COORD
    )

    part9 = p76.load_module(
        p76.P9,
        "part76_report_only_part9",
    )

    part11 = p76.load_module(
        p76.P11,
        "part76_report_only_part11",
    )

    print("")
    print(f"Device                 : {device}")
    print(f"Train cohort           : {len(train_rows)}")
    print(f"Validation cohort      : {len(val_rows)}")
    print(f"Full shape             : {p76.FULL}")
    print(f"Crop shape             : {p76.CROP}")
    print(f"Initialization SHA256  : {sha256_file(INIT_CKPT)}")
    print(f"Part 76 source         : {PART76_SOURCE}")
    print("")

    # ------------------------------------------------------------------------
    # Evaluate every existing checkpoint.
    # ------------------------------------------------------------------------
    trajectories = []
    class_trajectories = []
    case_trajectories = []
    point_trajectories = []

    for condition in CONDITIONS:
        banner(f"REPORT-ONLY CONDITION: {condition}")

        for epoch in range(1, EPOCHS + 1):
            ckpt = checkpoint_path(
                condition,
                epoch,
            )

            print(
                f"\nLoading checkpoint: {ckpt.name}"
            )

            model = p76.create_model(part11)

            payload = torch.load(
                ckpt,
                map_location="cpu",
            )

            if (
                isinstance(payload, dict)
                and "model_state_dict" in payload
            ):
                state = payload["model_state_dict"]
            else:
                state = payload

            model.load_state_dict(
                state,
                strict=True,
            )

            model.eval()

            loss_fn = p76.create_loss()

            (
                summary,
                class_summary,
                class_case_df,
                case_df,
                point_df,
            ) = p76.evaluate_model(
                model,
                loss_fn,
                val_rows,
                part9,
                part11,
                coord,
                condition,
            )

            # Add explicit epoch/condition labels.
            class_summary = class_summary.copy()
            class_summary["condition"] = condition
            class_summary["epoch"] = epoch

            class_case_df = class_case_df.copy()
            class_case_df["condition"] = condition
            class_case_df["epoch"] = epoch

            case_df = case_df.copy()
            case_df["condition"] = condition
            case_df["epoch"] = epoch

            point_df = point_df.copy()
            if len(point_df):
                point_df["condition"] = condition
                point_df["epoch"] = epoch

            trajectories.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "train_loss": np.nan,
                    "train_seg_loss": np.nan,
                    "train_point_loss": np.nan,
                    "train_mean_points": np.nan,
                    "train_target_fg": np.nan,
                    "val_loss": summary["loss"],
                    "val_fg_dice": summary[
                        "foreground_dice"
                    ],
                    "val_pred_fg": summary[
                        "foreground_predicted_voxels"
                    ],
                    "val_empty_fg": summary[
                        "foreground_empty_cases"
                    ],
                    "val_c2_c3_points": len(point_df),
                }
            )

            class_trajectories.append(
                class_summary
            )

            case_trajectories.append(
                class_case_df
            )

            point_trajectories.append(
                point_df
            )

            print(
                f"E{epoch} | "
                f"val_loss={summary['loss']:.6f} "
                f"FGDice={summary['foreground_dice']:.6f} "
                f"PredFG={summary['foreground_predicted_voxels']} "
                f"Empty={summary['foreground_empty_cases']}/{len(val_rows)} "
                f"C2/C3 points={len(point_df)}"
            )

            del model
            del loss_fn
            reset_cuda()

    trajectory_df = pd.DataFrame(
        trajectories
    )

    class_df = pd.concat(
        class_trajectories,
        ignore_index=True,
    )

    case_df = pd.concat(
        case_trajectories,
        ignore_index=True,
    )

    nonempty_points = [
        x
        for x in point_trajectories
        if len(x)
    ]

    point_df = (
        pd.concat(
            nonempty_points,
            ignore_index=True,
        )
        if nonempty_points
        else pd.DataFrame()
    )

    # ------------------------------------------------------------------------
    # Save regenerated trajectories.
    # ------------------------------------------------------------------------
    trajectory_df.to_csv(
        REPORT / "part76_learning_trajectory.csv",
        index=False,
    )

    class_df.to_csv(
        REPORT / "part76_classwise_trajectory.csv",
        index=False,
    )

    case_df.to_csv(
        REPORT / "part76_case_trajectory.csv",
        index=False,
    )

    point_df.to_csv(
        REPORT / "part76_point_trajectory.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Best epoch for each condition.
    # ------------------------------------------------------------------------
    best_rows = []

    for condition in CONDITIONS:
        g = trajectory_df[
            trajectory_df["condition"] == condition
        ].copy()

        idx = g["val_fg_dice"].idxmax()
        best_rows.append(
            g.loc[idx].to_dict()
        )

    best_df = pd.DataFrame(best_rows)

    baseline_best = best_df[
        best_df["condition"]
        == "R2_FULL_CLASS_BALANCED"
    ].iloc[0]

    aux_best = best_df[
        best_df["condition"]
        == "R2_FULL_CLASS_BALANCED_POINT_AUX"
    ].iloc[0]

    base_epoch = int(
        baseline_best["epoch"]
    )
    aux_epoch = int(
        aux_best["epoch"]
    )

    # ------------------------------------------------------------------------
    # Best-epoch class/localization reports.
    # ------------------------------------------------------------------------
    base_class = class_df[
        (class_df["condition"] == "R2_FULL_CLASS_BALANCED")
        & (class_df["epoch"] == base_epoch)
    ].copy()

    aux_class = class_df[
        (
            class_df["condition"]
            == "R2_FULL_CLASS_BALANCED_POINT_AUX"
        )
        & (class_df["epoch"] == aux_epoch)
    ].copy()

    base_points = point_df[
        (point_df["condition"] == "R2_FULL_CLASS_BALANCED")
        & (point_df["epoch"] == base_epoch)
    ].copy()

    aux_points = point_df[
        (
            point_df["condition"]
            == "R2_FULL_CLASS_BALANCED_POINT_AUX"
        )
        & (point_df["epoch"] == aux_epoch)
    ].copy()

    base_loc = make_c2_c3_localization(
        base_class,
        base_points,
        "R2_FULL_CLASS_BALANCED",
        base_epoch,
    )

    aux_loc = make_c2_c3_localization(
        aux_class,
        aux_points,
        "R2_FULL_CLASS_BALANCED_POINT_AUX",
        aux_epoch,
    )

    loc_df = pd.concat(
        [base_loc, aux_loc],
        ignore_index=True,
    )

    loc_df.to_csv(
        REPORT
        / "part76_best_epoch_c2_c3_localization.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Condition comparison.
    # ------------------------------------------------------------------------
    baseline_dice = float(
        baseline_best["val_fg_dice"]
    )
    aux_dice = float(
        aux_best["val_fg_dice"]
    )

    comparison = pd.DataFrame(
        [
            {
                "condition": baseline_best["condition"],
                "best_epoch": base_epoch,
                "best_val_fg_dice": baseline_dice,
                "best_val_loss": float(
                    baseline_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    baseline_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    baseline_best["val_empty_fg"]
                ),
                "delta_vs_baseline": 0.0,
            },
            {
                "condition": aux_best["condition"],
                "best_epoch": aux_epoch,
                "best_val_fg_dice": aux_dice,
                "best_val_loss": float(
                    aux_best["val_loss"]
                ),
                "best_val_pred_fg": float(
                    aux_best["val_pred_fg"]
                ),
                "best_val_empty_fg": int(
                    aux_best["val_empty_fg"]
                ),
                "delta_vs_baseline": (
                    aux_dice - baseline_dice
                ),
            },
        ]
    )

    comparison.to_csv(
        REPORT
        / "part76_condition_comparison.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # C2/C3 delta report.
    # ------------------------------------------------------------------------
    b = base_loc.set_index("class_id")
    a = aux_loc.set_index("class_id")

    delta_rows = []

    for c in (2, 3):
        delta_rows.append(
            {
                "class_id": c,
                "class_name": p76.CLASSES[c],
                "baseline_epoch": base_epoch,
                "point_aux_epoch": aux_epoch,
                "baseline_dice": float(
                    b.loc[c, "dice"]
                ),
                "point_aux_dice": float(
                    a.loc[c, "dice"]
                ),
                "delta_dice": float(
                    a.loc[c, "dice"]
                    - b.loc[c, "dice"]
                ),
                "baseline_precision": float(
                    b.loc[c, "precision"]
                ),
                "point_aux_precision": float(
                    a.loc[c, "precision"]
                ),
                "delta_precision": float(
                    a.loc[c, "precision"]
                    - b.loc[c, "precision"]
                ),
                "baseline_recall": float(
                    b.loc[c, "recall"]
                ),
                "point_aux_recall": float(
                    a.loc[c, "recall"]
                ),
                "delta_recall": float(
                    a.loc[c, "recall"]
                    - b.loc[c, "recall"]
                ),
                "baseline_prediction_target_ratio": float(
                    b.loc[
                        c,
                        "prediction_target_ratio",
                    ]
                ),
                "point_aux_prediction_target_ratio": float(
                    a.loc[
                        c,
                        "prediction_target_ratio",
                    ]
                ),
                "baseline_point_on_prediction": float(
                    b.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "point_aux_point_on_prediction": float(
                    a.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "delta_point_on_prediction": float(
                    a.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                    - b.loc[
                        c,
                        "point_on_prediction_fraction",
                    ]
                ),
                "baseline_annotation_to_prediction_distance": float(
                    b.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
                "point_aux_annotation_to_prediction_distance": float(
                    a.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
                "delta_annotation_to_prediction_distance": float(
                    a.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                    - b.loc[
                        c,
                        "annotation_to_prediction_mean_distance",
                    ]
                ),
            }
        )

    delta_df = pd.DataFrame(delta_rows)

    delta_df.to_csv(
        REPORT / "part76_c2_c3_delta.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Diagnosis.
    #
    # This uses the completed experiment, not the failed report stage.
    # ------------------------------------------------------------------------
    mean_focus_delta = float(
        delta_df["delta_dice"].mean()
    )

    mean_point_hit_delta = float(
        delta_df[
            "delta_point_on_prediction"
        ].mean()
    )

    finite_dist_delta = (
        pd.to_numeric(
            delta_df[
                "delta_annotation_to_prediction_distance"
            ],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    mean_distance_delta = (
        float(finite_dist_delta.mean())
        if len(finite_dist_delta)
        else np.nan
    )

    if (
        mean_focus_delta >= 0.02
        and mean_point_hit_delta >= 0.02
        and (
            np.isnan(mean_distance_delta)
            or mean_distance_delta <= 0
        )
    ):
        diagnosis = (
            "MEANINGFUL_POINT_AUXILIARY_C2_C3_LOCALIZATION_IMPROVEMENT"
        )
    elif mean_focus_delta >= 0.02:
        diagnosis = (
            "POINT_AUXILIARY_IMPROVES_C2_C3_DICE_BUT_LOCALIZATION_GAIN_IS_INCOMPLETE"
        )
    elif mean_point_hit_delta >= 0.02:
        diagnosis = (
            "POINT_AUXILIARY_IMPROVES_POINT_LOCALIZATION_WITHOUT_MEANINGFUL_DICE_GAIN"
        )
    elif (
        not np.isnan(mean_distance_delta)
        and mean_distance_delta < 0
    ):
        diagnosis = (
            "POINT_AUXILIARY_REDUCES_ANNOTATION_TO_PREDICTION_DISTANCE_WITHOUT_MEANINGFUL_DICE_GAIN"
        )
    else:
        diagnosis = (
            "NO_MEANINGFUL_C2_C3_POINT_AUXILIARY_ADVANTAGE"
        )

    # ------------------------------------------------------------------------
    # JSON summary.
    # ------------------------------------------------------------------------
    summary = {
        "part": 76,
        "mode": "REPORT_ONLY",
        "training_performed": False,
        "optimizer_steps_performed": False,
        "backward_pass_performed": False,
        "checkpoint_modified": False,
        "device": str(device),
        "train_cases": TRAIN_N,
        "validation_cases": VAL_N,
        "epochs_evaluated": EPOCHS,
        "full_shape": list(p76.FULL),
        "crop_shape": list(p76.CROP),
        "seed": SEED,
        "focus_classes": [2, 3],
        "point_aux_lambda": float(
            p76.POINT_AUX_LAMBDA
        ),
        "initialization_checkpoint": str(
            INIT_CKPT
        ),
        "initialization_sha256": sha256_file(
            INIT_CKPT
        ),
        "baseline_best": baseline_best.to_dict(),
        "point_aux_best": aux_best.to_dict(),
        "overall_dice_delta": (
            aux_dice - baseline_dice
        ),
        "mean_c2_c3_dice_delta": mean_focus_delta,
        "mean_point_hit_delta": mean_point_hit_delta,
        "mean_annotation_to_prediction_distance_delta": (
            mean_distance_delta
        ),
        "diagnosis": diagnosis,
        "spider_used": False,
        "rsna_test_set_used": False,
        "point_annotations_are_segmentation_ground_truth": False,
        "scientific_limitation": (
            "RSNA coordinates are point/localizer annotations, "
            "not manual segmentation masks. Point supervision "
            "therefore represents localization/target-consistency "
            "supervision rather than clinical segmentation ground truth."
        ),
        "note_on_training_columns": (
            "The original Part 76 checkpoints store model and optimizer "
            "states but do not store per-epoch training-loss summaries. "
            "Therefore the report-only learning trajectory contains "
            "validation metrics and leaves training-loss fields NaN."
        ),
    }

    with open(
        REPORT / "part76_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=True,
        )

    # ------------------------------------------------------------------------
    # Text report.
    # ------------------------------------------------------------------------
    lines = [
        "PART 76 — REPORT-ONLY RECONSTRUCTION",
        "",
        "No training was performed during report generation.",
        "Existing Part 76 checkpoints were evaluated only.",
        "",
        f"Device: {device}",
        f"Train cohort: {TRAIN_N}",
        f"Validation cohort: {VAL_N}",
        f"Full shape: {p76.FULL}",
        f"Crop shape: {p76.CROP}",
        "",
        "BASELINE — R2_FULL_CLASS_BALANCED",
        f"Best epoch: {base_epoch}",
        f"Best foreground Dice: {baseline_dice:.6f}",
        "",
        "POINT AUX — R2_FULL_CLASS_BALANCED_POINT_AUX",
        f"Best epoch: {aux_epoch}",
        f"Best foreground Dice: {aux_dice:.6f}",
        "",
        f"Overall Dice delta: {aux_dice - baseline_dice:+.6f}",
        f"Mean C2/C3 Dice delta: {mean_focus_delta:+.6f}",
        f"Mean point-on-prediction delta: {mean_point_hit_delta:+.6f}",
        f"Mean annotation->prediction distance delta: {mean_distance_delta:+.6f}",
        "",
        "C2/C3 BEST-EPOCH COMPARISON",
    ]

    for _, r in delta_df.iterrows():
        lines.extend(
            [
                "",
                f"C{int(r['class_id'])} {r['class_name']}",
                (
                    f"  Dice: "
                    f"{r['baseline_dice']:.6f} -> "
                    f"{r['point_aux_dice']:.6f} "
                    f"(delta {r['delta_dice']:+.6f})"
                ),
                (
                    f"  Recall: "
                    f"{r['baseline_recall']:.6f} -> "
                    f"{r['point_aux_recall']:.6f} "
                    f"(delta {r['delta_recall']:+.6f})"
                ),
                (
                    f"  Precision: "
                    f"{r['baseline_precision']:.6f} -> "
                    f"{r['point_aux_precision']:.6f} "
                    f"(delta {r['delta_precision']:+.6f})"
                ),
                (
                    f"  Prediction/target ratio: "
                    f"{r['baseline_prediction_target_ratio']:.3f} -> "
                    f"{r['point_aux_prediction_target_ratio']:.3f}"
                ),
                (
                    f"  Point-on-prediction: "
                    f"{r['baseline_point_on_prediction']:.6f} -> "
                    f"{r['point_aux_point_on_prediction']:.6f}"
                ),
                (
                    f"  Annotation->prediction distance: "
                    f"{r['baseline_annotation_to_prediction_distance']:.3f} -> "
                    f"{r['point_aux_annotation_to_prediction_distance']:.3f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            f"DIAGNOSIS: {diagnosis}",
            "",
            "SCIENTIFIC LIMITATION:",
            "RSNA coordinates are point/localizer annotations, not",
            "manual segmentation masks. Point agreement is therefore",
            "a localization/target-consistency diagnostic and not",
            "medical segmentation accuracy.",
            "",
            "REPORT-ONLY NOTE:",
            "Training-loss columns cannot be reconstructed from the",
            "existing checkpoints because the original checkpoints",
            "did not store the per-epoch training-loss summaries.",
        ]
    )

    (
        REPORT / "part76_report.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Final console summary.
    # ------------------------------------------------------------------------
    banner("PART 76 REPORT-ONLY COMPLETE")

    print(
        f"Baseline best Dice       : "
        f"{baseline_dice:.6f} (E{base_epoch})"
    )
    print(
        f"Point-aux best Dice      : "
        f"{aux_dice:.6f} (E{aux_epoch})"
    )
    print(
        f"Overall Dice delta       : "
        f"{aux_dice - baseline_dice:+.6f}"
    )
    print(
        f"Mean C2/C3 Dice delta    : "
        f"{mean_focus_delta:+.6f}"
    )
    print(
        f"Mean point-hit delta     : "
        f"{mean_point_hit_delta:+.6f}"
    )
    print(
        f"Mean annotation->pred Δ  : "
        f"{mean_distance_delta:+.6f}"
    )
    print(
        f"Diagnosis                 : "
        f"{diagnosis}"
    )

    print("")
    print("Generated reports:")

    for p in [
        REPORT / "part76_learning_trajectory.csv",
        REPORT / "part76_classwise_trajectory.csv",
        REPORT / "part76_case_trajectory.csv",
        REPORT / "part76_point_trajectory.csv",
        REPORT / "part76_best_epoch_c2_c3_localization.csv",
        REPORT / "part76_c2_c3_delta.csv",
        REPORT / "part76_condition_comparison.csv",
        REPORT / "part76_summary.json",
        REPORT / "part76_report.txt",
    ]:
        print(f"  {p}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("PART 76 REPORT-ONLY FAILED")
        traceback.print_exc()
        raise
