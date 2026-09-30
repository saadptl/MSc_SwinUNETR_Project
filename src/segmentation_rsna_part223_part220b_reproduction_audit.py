"""
PART 2.23 — PART 2.20B EXACT REPRODUCTION AUDIT
================================================

Purpose
-------
Reproduce the Part 2.20B Epoch-5 validation pipeline as literally as
possible using the actual Part 2.20B source implementation.

This audit exists because Part 2.22 produced 18.62% point accuracy while
Part 2.20B reported 41.49%. Before any further model training, we must
determine whether the difference is caused by an evaluation-pipeline
mismatch.

This script:
- imports the actual Part 2.20B source
- uses Part 2.20B's own validation-study selection
- uses Part 2.20B's own load_case()
- uses Part 2.20B's own geometry/resampling/normalization
- uses Part 2.20B's own evaluate_case()
- loads the saved Part 2.20B Epoch-5 checkpoint
- reproduces the point-level validation metrics
- additionally records point-by-point predictions
- compares against the historical Part 2.20B Epoch-5 result

NO TRAINING.
NO BACKWARD().
NO OPTIMIZER STEP.
NO CHECKPOINT MODIFICATION.
NO DASHBOARD MODIFICATION.
NO VOXEL GROUND TRUTH.

Important:
RSNA annotations are point/localization annotations, not manual
voxel-wise segmentation masks. Accuracy here is point-level only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

PART220B_PATH = (
    SRC / "segmentation_rsna_part220b_geometry_corrected_training.py"
)

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
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints"
    / "part220b_epoch_05.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part223_part220b_reproduction_audit"
)

REPORT_DIR = OUTPUT_DIR / "reports"


# ============================================================
# 2. EXPECTED HISTORICAL RESULT
# ============================================================

EXPECTED_PART220B_EPOCH5 = {
    "overall_accuracy": 0.4148936170212766,
    "macro_disease_accuracy": 0.444448864989634,
    "mean_true_probability": 0.287157,
    "hit_rate_0_50": 0.106383,
}


# ============================================================
# 3. LOAD ACTUAL PART 2.20B MODULE
# ============================================================

def load_part220b():
    if not PART220B_PATH.exists():
        raise FileNotFoundError(
            f"Actual Part 2.20B source was not found:\n{PART220B_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "part220b_exact_source",
        PART220B_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create import specification for Part 2.20B."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# 4. LOAD CHECKPOINT
# ============================================================

def load_checkpoint_exact(module):
    device = module.DEVICE

    model = module.build_model()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Part 2.20B Epoch-5 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint.get("model", checkpoint),
        ),
    )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Could not locate model state_dict in checkpoint."
        )

    cleaned = {}

    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    result = model.load_state_dict(
        cleaned,
        strict=True,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model.eval()

    return model, checkpoint


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 78)
    print("PART 2.23 — PART 2.20B EXACT REPRODUCTION AUDIT")
    print("=" * 78)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Project root: {ROOT}")
    print(f"Part 2.20B source: {PART220B_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # Load actual Part 2.20B implementation.
    # --------------------------------------------------------
    print("\nLoading the ACTUAL Part 2.20B implementation...")
    p220b = load_part220b()

    print("Part 2.20B module loaded successfully.")

    # --------------------------------------------------------
    # Verify the functions we intend to use.
    # --------------------------------------------------------
    required_functions = [
        "load_manifest",
        "select_validation_series",
        "build_case_index",
        "load_case",
        "evaluate_case",
        "build_model",
    ]

    missing_functions = [
        name
        for name in required_functions
        if not hasattr(p220b, name)
    ]

    if missing_functions:
        raise RuntimeError(
            "Actual Part 2.20B source is missing required functions: "
            f"{missing_functions}"
        )

    print("\nExact Part 2.20B functions available:")
    for name in required_functions:
        print(f"  OK: {name}")

    # --------------------------------------------------------
    # Manifest.
    # --------------------------------------------------------
    manifest = p220b.load_manifest()

    print(f"\nManifest rows: {len(manifest)}")
    print(
        "Annotated series:",
        manifest[["study_id", "series_id"]]
        .drop_duplicates()
        .shape[0],
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Use Part 2.20B's EXACT validation selector.
    # --------------------------------------------------------
    selected = p220b.select_validation_series(manifest)

    print(
        f"Part 2.20B validation series selected: {len(selected)}"
    )
    print(
        "Part 2.20B validation studies selected:",
        selected["study_id"].astype(str).nunique(),
    )

    selected.to_csv(
        OUTPUT_DIR / "part223_exact_validation_series.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Display the exact selected cohort.
    # --------------------------------------------------------
    print("\nExact Part 2.20B validation cohort:")
    print(
        selected[
            ["study_id", "series_id"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Build validation cases EXACTLY like Part 2.20B.
    # --------------------------------------------------------
    validation_keys = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        )
        for _, row in selected.iterrows()
    }

    validation_manifest = manifest[
        manifest.apply(
            lambda row:
            (
                str(row["study_id"]),
                str(row["series_id"]),
            ) in validation_keys,
            axis=1,
        )
    ].copy()

    validation_cases = p220b.build_case_index(
        validation_manifest
    )

    # Preserve exact selected ordering.
    case_order = {
        (
            str(row["study_id"]),
            str(row["series_id"]),
        ): index
        for index, row in selected.iterrows()
    }

    validation_cases = sorted(
        validation_cases,
        key=lambda case:
        case_order.get(
            (
                str(case["study_id"]),
                str(case["series_id"]),
            ),
            10_000,
        ),
    )

    print(
        f"\nValidation cases constructed: {len(validation_cases)}"
    )

    # --------------------------------------------------------
    # Model and checkpoint.
    # --------------------------------------------------------
    print("\nLoading Part 2.20B Epoch-5 checkpoint...")
    model, checkpoint = load_checkpoint_exact(p220b)

    print("Part 2.20B Epoch-5 checkpoint loaded successfully.")

    checkpoint_epoch = checkpoint.get("epoch")
    print(f"Checkpoint epoch: {checkpoint_epoch}")

    # --------------------------------------------------------
    # Exact evaluation.
    #
    # This uses the actual Part 2.20B load_case() and
    # evaluate_case() without reproducing them independently.
    # --------------------------------------------------------
    all_rows = []
    foreground_ratios = []
    failures = []

    print("\n" + "=" * 78)
    print("EXACT PART 2.20B VALIDATION REPRODUCTION")
    print("=" * 78)

    for index, case in enumerate(validation_cases, start=1):
        study_id = str(case["study_id"])
        series_id = str(case["series_id"])

        print(
            f"\n[{index:02d}/{len(validation_cases):02d}] "
            f"{study_id}/{series_id}"
        )

        try:
            # EXACT Part 2.20B loading.
            image, points, geometry = p220b.load_case(
                study_id,
                series_id,
                case["points"],
            )

            # EXACT Part 2.20B evaluation.
            rows, foreground = p220b.evaluate_case(
                model,
                image,
                points,
            )

            foreground_ratios.append(
                float(foreground)
            )

            print(f"  Points: {len(rows)}")
            print(
                f"  Foreground ratio: "
                f"{float(foreground):.6f}"
            )

            for point_row, point in zip(rows, points):
                row = dict(point_row)

                row["study_id"] = study_id
                row["series_id"] = series_id
                row["series_description"] = str(
                    case["points"].iloc[0].get(
                        "series_description",
                        "",
                    )
                )

                # Preserve physical/canonical coordinates from the
                # actual Part 2.20B load_case().
                row["patient_x"] = float(
                    point.get("patient_x", np.nan)
                )
                row["patient_y"] = float(
                    point.get("patient_y", np.nan)
                )
                row["patient_z"] = float(
                    point.get("patient_z", np.nan)
                )
                row["canonical_z"] = float(
                    point.get("z", np.nan)
                )
                row["canonical_y"] = float(
                    point.get("y", np.nan)
                )
                row["canonical_x"] = float(
                    point.get("x", np.nan)
                )

                all_rows.append(row)

        except Exception as exc:
            print(
                f"  FAILED: {type(exc).__name__}: {exc}"
            )

            failures.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    results = pd.DataFrame(all_rows)

    if results.empty:
        raise RuntimeError(
            "Exact Part 2.20B reproduction produced no results."
        )

    # --------------------------------------------------------
    # Calculate the SAME metrics as Part 2.20B validate().
    # --------------------------------------------------------
    disease_accuracy = (
        results[
            results["true_class"] > 0
        ]
        .groupby("true_class")["correct"]
        .mean()
    )

    overall_accuracy = float(
        results["correct"].mean()
    )

    macro_accuracy = float(
        disease_accuracy.mean()
    ) if len(disease_accuracy) else 0.0

    mean_probability = float(
        results["true_probability"].mean()
    )

    hit_rate = float(
        (
            results["true_probability"] >= 0.50
        ).mean()
    )

    mean_foreground_ratio = float(
        np.mean(foreground_ratios)
    ) if foreground_ratios else 0.0

    # --------------------------------------------------------
    # Disease-wise metrics.
    # --------------------------------------------------------
    disease_rows = []

    for class_id in range(1, p220b.NUM_CLASSES):
        subset = results[
            results["true_class"] == class_id
        ]

        if subset.empty:
            continue

        disease_rows.append(
            {
                "class_id": class_id,
                "class_name": p220b.CLASS_NAMES[class_id],
                "points": int(len(subset)),
                "accuracy": float(
                    subset["correct"].mean()
                ),
                "mean_true_probability": float(
                    subset["true_probability"].mean()
                ),
                "hit_rate_0_50": float(
                    (
                        subset["true_probability"] >= 0.50
                    ).mean()
                ),
            }
        )

    disease_df = pd.DataFrame(disease_rows)

    # --------------------------------------------------------
    # Compare against historical Part 2.20B Epoch 5.
    # --------------------------------------------------------
    comparisons = []

    current_metrics = {
        "overall_accuracy": overall_accuracy,
        "macro_disease_accuracy": macro_accuracy,
        "mean_true_probability": mean_probability,
        "hit_rate_0_50": hit_rate,
    }

    for metric_name, expected in EXPECTED_PART220B_EPOCH5.items():
        actual = current_metrics[metric_name]
        delta = actual - expected

        # Accuracy metrics are deterministic enough that tiny
        # floating-point differences are acceptable.
        matches = bool(abs(delta) <= 1e-6)

        comparisons.append(
            {
                "metric": metric_name,
                "historical_part220b_epoch5": expected,
                "reproduced_part220b": actual,
                "delta": delta,
                "matches_within_1e-6": matches,
            }
        )

    comparison_df = pd.DataFrame(comparisons)

    exact_match = bool(
        comparison_df["matches_within_1e-6"].all()
    )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------
    results.to_csv(
        OUTPUT_DIR / "part223_exact_point_results.csv",
        index=False,
    )

    disease_df.to_csv(
        OUTPUT_DIR / "part223_exact_disease_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "part223_historical_metric_comparison.csv",
        index=False,
    )

    pd.DataFrame(failures).to_csv(
        OUTPUT_DIR / "part223_failed_cases.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary JSON.
    # --------------------------------------------------------
    summary = {
        "part": "2.23",
        "status": "COMPLETE",
        "purpose": "Exact Part 2.20B validation reproduction audit",
        "part220b_source": str(PART220B_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint_epoch,
        "validation_series": int(len(selected)),
        "validation_studies": int(
            selected["study_id"].astype(str).nunique()
        ),
        "processed_cases": int(
            len(validation_cases) - len(failures)
        ),
        "failed_cases": int(len(failures)),
        "audited_points": int(len(results)),
        "overall_accuracy": overall_accuracy,
        "macro_disease_accuracy": macro_accuracy,
        "mean_true_probability": mean_probability,
        "hit_rate_0_50": hit_rate,
        "mean_foreground_ratio": mean_foreground_ratio,
        "historical_part220b_epoch5": EXPECTED_PART220B_EPOCH5,
        "exact_metric_match_within_1e-6": exact_match,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "manual_voxel_ground_truth_created": False,
        "scientific_limitation": (
            "RSNA coordinates are point/localization annotations, "
            "not manual voxel-wise segmentation masks. "
            "Metrics are point-level prediction metrics only."
        ),
    }

    with open(
        OUTPUT_DIR / "part223_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Human-readable report.
    # --------------------------------------------------------
    report_lines = [
        "=" * 78,
        "PART 2.23 — PART 2.20B EXACT REPRODUCTION AUDIT",
        "=" * 78,
        "",
        f"Part 2.20B source: {PART220B_PATH}",
        f"Checkpoint: {CHECKPOINT_PATH}",
        f"Checkpoint epoch: {checkpoint_epoch}",
        "",
        f"Validation series: {len(selected)}",
        f"Processed cases: {len(validation_cases) - len(failures)}",
        f"Failed cases: {len(failures)}",
        f"Audited points: {len(results)}",
        "",
        "REPRODUCED METRICS",
        "-" * 78,
        f"Overall point accuracy: {overall_accuracy:.12f}",
        f"Macro disease accuracy: {macro_accuracy:.12f}",
        f"Mean true probability: {mean_probability:.12f}",
        f"Hit rate @ 0.50: {hit_rate:.12f}",
        f"Mean foreground ratio: {mean_foreground_ratio:.12f}",
        "",
        "HISTORICAL VS REPRODUCED",
        "-" * 78,
        comparison_df.to_string(index=False),
        "",
        f"Exact metric match within 1e-6: {exact_match}",
        "",
        "DISEASE SUMMARY",
        "-" * 78,
        disease_df.to_string(index=False),
        "",
        "SCIENTIFIC LIMITATION",
        "-" * 78,
        summary["scientific_limitation"],
        "",
        "SAFETY",
        "-" * 78,
        "Training performed: NO",
        "Backward performed: NO",
        "Optimizer step performed: NO",
        "Checkpoint modified: NO",
        "Dashboard modified: NO",
        "Manual voxel ground truth created: NO",
    ]

    with open(
        REPORT_DIR
        / "part223_part220b_reproduction_audit_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(report_lines))

    # --------------------------------------------------------
    # Terminal summary.
    # --------------------------------------------------------
    print("\n")
    print("=" * 78)
    print("PART 2.23 COMPLETE")
    print("=" * 78)

    print(
        f"Processed cases : "
        f"{len(validation_cases) - len(failures)}/"
        f"{len(validation_cases)}"
    )

    print(
        f"Audited points  : {len(results)}"
    )

    print(
        f"Overall accuracy: {overall_accuracy:.12f}"
    )

    print(
        f"Macro accuracy  : {macro_accuracy:.12f}"
    )

    print(
        f"Mean probability: {mean_probability:.12f}"
    )

    print(
        f"Hit rate @0.50  : {hit_rate:.12f}"
    )

    print(
        f"Mean foreground : {mean_foreground_ratio:.12f}"
    )

    print(
        f"Historical match: {'YES' if exact_match else 'NO'}"
    )

    rfnn_to_lfnn = int(
        (
            (results["true_class"] == 3)
            & (results["predicted_class"] == 2)
        ).sum()
    )

    lfnn_to_rfnn = int(
        (
            (results["true_class"] == 2)
            & (results["predicted_class"] == 3)
        ).sum()
    )

    print(
        f"RFNN -> LFNN    : {rfnn_to_lfnn}"
    )

    print(
        f"LFNN -> RFNN    : {lfnn_to_rfnn}"
    )

    print("")
    print(f"Outputs: {OUTPUT_DIR}")
    print("")
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
