from pathlib import Path
import sys
import json
import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import segmentation_part33_balanced_symmetry_refinement as p33


CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_balanced_symmetry_refinement"
    / "checkpoints"
    / "part33_best_development_macro.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_untouched_test_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("PART 3.3 — UNTOUCHED TEST EVALUATION")
    print("=" * 80)
    print()
    print("READ-ONLY EVALUATION")
    print("No training.")
    print("No checkpoint modification.")
    print("No dashboard modification.")
    print("No fabricated voxel masks.")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Part3.3 checkpoint not found:\n{CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = p33.part220b.load_manifest()

    print(f"Manifest rows: {len(manifest)}")

    # ------------------------------------------------------------------
    # EXACT Part3.3 validation cohort
    # ------------------------------------------------------------------
    validation_cases = p33.build_validation_sets(manifest)

    validation_keys = {
        (
            str(case["study_id"]),
            str(case["series_id"]),
        )
        for case in validation_cases
    }

    print(f"Development cases: {len(validation_cases)}")

    # ------------------------------------------------------------------
    # EXACT Part3.3 test-cohort construction
    # ------------------------------------------------------------------
    all_cases = p33.part220b.build_case_index(manifest)

    remaining_cases = [
        case
        for case in all_cases
        if (
            str(case["study_id"]),
            str(case["series_id"]),
        ) not in validation_keys
    ]

    remaining_cases = sorted(
        remaining_cases,
        key=lambda case: (
            str(case["study_id"]),
            str(case["series_id"]),
        ),
    )

    test_cases = remaining_cases[:25]

    print(f"Untouched test cases: {len(test_cases)}")

    if len(test_cases) != 25:
        raise RuntimeError(
            f"Expected 25 untouched test cases, got {len(test_cases)}."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = p33.part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    state_dict = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch. "
            f"Missing={len(missing)}, Unexpected={len(unexpected)}"
        )

    model = model.to(device)
    model.eval()

    print("Checkpoint loaded cleanly.")
    print(f"Checkpoint: {CHECKPOINT}")

    print()
    print("=" * 80)
    print("RUNNING UNTOUCHED TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # EXACT SAME evaluate_cases() USED BY PART3.3
    # ------------------------------------------------------------------
    test = p33.evaluate_cases(
        model,
        test_cases,
        device,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("PART 3.3 TEST RESULTS")
    print("=" * 80)

    print(f"Test cases: {len(test_cases)}")
    print(f"Test points: {test['points']}")
    print(f"Test overall: {test['overall']:.6f}")
    print(f"Test macro disease: {test['macro']:.6f}")
    print(
        f"Test mean true probability: "
        f"{test['mean_probability']:.6f}"
    )
    print(
        f"Test hit @0.50: "
        f"{test['hit_050']:.6f}"
    )
    print(
        f"Test foreground ratio: "
        f"{test['foreground_ratio']:.6f}"
    )

    print()
    print("Disease-wise test accuracy:")

    for disease in test["disease"]:
        print(
            f"  {disease['class_name']}: "
            f"{disease['accuracy']:.6f}"
        )

    # ------------------------------------------------------------------
    # Save records
    # ------------------------------------------------------------------
    records_path = (
        OUTPUT_DIR
        / "part33_test_point_records.csv"
    )

    disease_path = (
        OUTPUT_DIR
        / "part33_test_disease_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "part33_untouched_test_summary.json"
    )

    pd.DataFrame(
        test["records"]
    ).to_csv(
        records_path,
        index=False,
    )

    pd.DataFrame(
        test["disease"]
    ).to_csv(
        disease_path,
        index=False,
    )

    summary = {
        "experiment": "Part3.3",
        "evaluation_type": "untouched_test",
        "checkpoint": str(CHECKPOINT),
        "test_cases": len(test_cases),
        "test_points": test["points"],
        "test_overall": test["overall"],
        "test_macro": test["macro"],
        "test_mean_probability": test["mean_probability"],
        "test_hit_050": test["hit_050"],
        "test_foreground_ratio": test["foreground_ratio"],
        "disease": test["disease"],
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "manual_voxel_masks_fabricated": False,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Summary: {summary_path}")
    print(f"Records: {records_path}")
    print(f"Disease metrics: {disease_path}")


if __name__ == "__main__":
    main()
