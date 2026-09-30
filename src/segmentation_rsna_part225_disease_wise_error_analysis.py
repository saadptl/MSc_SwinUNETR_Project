"""
PART 2.25 — DISEASE-WISE ERROR ANALYSIS
=======================================

Uses the ACTUAL Part 2.20B implementation and Epoch-5 checkpoint.

Purpose:
- reproduce the exact Part 2.20B validation predictions
- analyze disease-wise performance
- analyze confusion
- analyze level-wise performance
- analyze confidence
- identify exact point-level errors
- inspect spatial coordinates without creating voxel ground truth

NO TRAINING
NO CHECKPOINT MODIFICATION
NO DASHBOARD MODIFICATION
NO FABRICATED VOXEL GROUND TRUTH
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PART220B = SRC / "segmentation_rsna_part220b_geometry_corrected_training.py"
CHECKPOINT = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints" / "part220b_epoch_05.pth"
)
OUT = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part225_disease_wise_error_analysis"
)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def load_part220b():
    spec = importlib.util.spec_from_file_location("part220b_exact", PART220B)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {PART220B}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part220b_exact"] = module
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PART 2.25 — DISEASE-WISE ERROR ANALYSIS")
    print("=" * 78)
    print(f"Part 2.20B source: {PART220B}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not PART220B.exists():
        raise FileNotFoundError(PART220B)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    part = load_part220b()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    manifest = part.load_manifest()
    selected = part.select_validation_series(manifest)

    keys = {
        (str(r["study_id"]), str(r["series_id"]))
        for _, r in selected.iterrows()
    }

    validation_manifest = manifest[
        manifest.apply(
            lambda r: (str(r["study_id"]), str(r["series_id"])) in keys,
            axis=1,
        )
    ].copy()

    cases = part.build_case_index(validation_manifest)

    model = part.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )
    state = (
        checkpoint.get("model_state_dict")
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint.get("state_dict")
        if isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        else checkpoint
    )

    result = model.load_state_dict(state, strict=True)
    print("Checkpoint loaded successfully.")
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")

    model.to(device)
    model.eval()

    all_rows = []
    case_rows = []

    for idx, case in enumerate(cases, start=1):
        study_id = str(case["study_id"])
        series_id = str(case["series_id"])

        print(f"[{idx:02d}/{len(cases)}] {study_id}/{series_id}")

        image, points, geometry = part.load_case(
            study_id,
            series_id,
            case["points"],
        )

        rows, foreground_ratio = part.evaluate_case(
            model,
            image,
            points,
        )

        for row in rows:
            row = dict(row)
            row["study_id"] = study_id
            row["series_id"] = series_id
            row["foreground_ratio"] = float(foreground_ratio)
            row["true_name"] = CLASS_NAMES.get(
                int(row["true_class"]), str(row["true_class"])
            )
            row["predicted_name"] = CLASS_NAMES.get(
                int(row["predicted_class"]), str(row["predicted_class"])
            )
            row["correct"] = int(
                int(row["true_class"]) == int(row["predicted_class"])
            )
            all_rows.append(row)

        case_rows.append(
            {
                "study_id": study_id,
                "series_id": series_id,
                "points": len(rows),
                "correct": sum(int(r["correct"]) for r in rows),
                "accuracy": (
                    np.mean([r["correct"] for r in rows])
                    if rows else np.nan
                ),
                "foreground_ratio": float(foreground_ratio),
            }
        )

    results = pd.DataFrame(all_rows)
    cases_df = pd.DataFrame(case_rows)

    if results.empty:
        raise RuntimeError("No prediction results were produced.")

    # ------------------------------------------------------------
    # Point-level error table
    # ------------------------------------------------------------
    error_cols = [
        "study_id",
        "series_id",
        "level",
        "true_class",
        "true_name",
        "predicted_class",
        "predicted_name",
        "correct",
        "true_probability",
        "predicted_probability",
        "z",
        "y",
        "x",
        "foreground_ratio",
    ]
    error_cols = [c for c in error_cols if c in results.columns]
    errors = results[error_cols].copy()
    errors["error_type"] = np.where(
        errors["correct"].astype(int) == 1,
        "Correct",
        errors["true_name"] + " -> " + errors["predicted_name"],
    )
    errors.to_csv(OUT / "point_error_analysis.csv", index=False)

    # ------------------------------------------------------------
    # Disease-wise metrics
    # ------------------------------------------------------------
    disease_rows = []
    for cid in range(1, 6):
        sub = results[results["true_class"].astype(int) == cid]
        if sub.empty:
            continue

        disease_rows.append(
            {
                "class_id": cid,
                "disease": CLASS_NAMES[cid],
                "points": len(sub),
                "accuracy": float(sub["correct"].mean()),
                "mean_true_probability": float(
                    sub["true_probability"].mean()
                ),
                "median_true_probability": float(
                    sub["true_probability"].median()
                ),
                "hit_rate_at_0_50": float(
                    (sub["true_probability"] >= 0.50).mean()
                ),
                "mean_foreground_ratio": float(
                    sub["foreground_ratio"].mean()
                ),
            }
        )

    disease_df = pd.DataFrame(disease_rows)
    disease_df.to_csv(OUT / "disease_metrics.csv", index=False)

    # ------------------------------------------------------------
    # Level-wise metrics
    # ------------------------------------------------------------
    level_rows = []
    for level in LEVELS:
        sub = results[results["level"].astype(str) == level]
        if sub.empty:
            continue

        level_rows.append(
            {
                "level": level,
                "points": len(sub),
                "accuracy": float(sub["correct"].mean()),
                "mean_true_probability": float(
                    sub["true_probability"].mean()
                ),
            }
        )

    level_df = pd.DataFrame(level_rows)
    level_df.to_csv(OUT / "level_metrics.csv", index=False)

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    confusion = pd.crosstab(
        results["true_name"],
        results["predicted_name"],
        dropna=False,
    )

    confusion.to_csv(OUT / "confusion_matrix.csv")

    # ------------------------------------------------------------
    # Disease confusion pairs
    # ------------------------------------------------------------
    wrong = results[results["correct"].astype(int) == 0].copy()
    if not wrong.empty:
        pair_df = (
            wrong.groupby(["true_name", "predicted_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        pair_df = pd.DataFrame(
            columns=["true_name", "predicted_name", "count"]
        )

    pair_df.to_csv(OUT / "confusion_pairs.csv", index=False)

    # ------------------------------------------------------------
    # Per-case summary
    # ------------------------------------------------------------
    cases_df.to_csv(OUT / "case_summary.csv", index=False)

    # ------------------------------------------------------------
    # Global metrics
    # ------------------------------------------------------------
    disease_accuracy = (
        results.groupby("true_class")["correct"]
        .mean()
        .reindex([1, 2, 3, 4, 5])
    )

    overall = float(results["correct"].mean())
    macro = float(disease_accuracy.mean())
    mean_prob = float(results["true_probability"].mean())
    hit = float((results["true_probability"] >= 0.50).mean())

    # ------------------------------------------------------------
    # Confidence/error analysis
    # ------------------------------------------------------------
    correct_df = results[results["correct"] == 1]
    incorrect_df = results[results["correct"] == 0]

    confidence_summary = {
        "correct_points": int(len(correct_df)),
        "incorrect_points": int(len(incorrect_df)),
        "correct_mean_true_probability": (
            float(correct_df["true_probability"].mean())
            if not correct_df.empty else None
        ),
        "incorrect_mean_true_probability": (
            float(incorrect_df["true_probability"].mean())
            if not incorrect_df.empty else None
        ),
        "incorrect_high_confidence_count": int(
            (
                incorrect_df["predicted_probability"] >= 0.80
            ).sum()
        ) if not incorrect_df.empty else 0,
    }

    # ------------------------------------------------------------
    # Exact confusion counts of the two foraminal classes
    # ------------------------------------------------------------
    rfnn_to_lfnn = int(
        (
            (results["true_class"].astype(int) == 3)
            & (results["predicted_class"].astype(int) == 2)
        ).sum()
    )
    lfnn_to_rfnn = int(
        (
            (results["true_class"].astype(int) == 2)
            & (results["predicted_class"].astype(int) == 3)
        ).sum()
    )

    summary = {
        "part": "2.25",
        "status": "COMPLETE",
        "validation_cases": int(len(cases)),
        "audited_points": int(len(results)),
        "overall_accuracy": overall,
        "macro_disease_accuracy": macro,
        "mean_true_probability": mean_prob,
        "hit_rate_at_0_50": hit,
        "rfnn_to_lfnn": rfnn_to_lfnn,
        "lfnn_to_rfnn": lfnn_to_rfnn,
        "confidence": confidence_summary,
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "voxel_ground_truth_fabricated": False,
        "interpretation_note": (
            "Metrics are point-supervision classification metrics. "
            "They are not voxel-wise Dice or clinical segmentation accuracy."
        ),
    }

    (OUT / "part225_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "PART 2.25 — DISEASE-WISE ERROR ANALYSIS",
        "",
        f"Validation cases: {len(cases)}",
        f"Audited points: {len(results)}",
        f"Overall point accuracy: {overall:.12f}",
        f"Macro disease accuracy: {macro:.12f}",
        f"Mean true-class probability: {mean_prob:.12f}",
        f"Hit rate @0.50: {hit:.12f}",
        "",
        "DISEASE METRICS",
        disease_df.to_string(index=False),
        "",
        "LEVEL METRICS",
        level_df.to_string(index=False),
        "",
        "CONFUSION MATRIX",
        confusion.to_string(),
        "",
        "TOP CONFUSION PAIRS",
        pair_df.head(15).to_string(index=False),
        "",
        f"RFNN -> LFNN: {rfnn_to_lfnn}",
        f"LFNN -> RFNN: {lfnn_to_rfnn}",
        "",
        "CONFIDENCE ANALYSIS",
        json.dumps(confidence_summary, indent=2),
        "",
        "Training performed: NO",
        "Checkpoint modified: NO",
        "Dashboard modified: NO",
        "Voxel ground truth fabricated: NO",
    ]

    (OUT / "part225_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PART 2.25 COMPLETE")
    print("=" * 78)
    print(f"Processed cases : {len(cases)}/{len(cases)}")
    print(f"Audited points  : {len(results)}")
    print(f"Overall accuracy: {overall:.12f}")
    print(f"Macro accuracy  : {macro:.12f}")
    print(f"Mean probability: {mean_prob:.12f}")
    print(f"Hit rate @0.50  : {hit:.12f}")
    print(f"RFNN -> LFNN    : {rfnn_to_lfnn}")
    print(f"LFNN -> RFNN    : {lfnn_to_rfnn}")
    print(f"Outputs         : {OUT}")
    print()
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
