"""
PART 123 — PART122 BEST CHECKPOINT FULL-VALIDATION INFERENCE + DISTRIBUTION AUDIT

Purpose
-------
Run the established Part113 full-validation inference contract using the
Part122 BEST checkpoint, then run the established Part114 prediction
distribution/spatial-quality audit on those new predictions.

This is an evaluation-only experiment:
- Part104 final checkpoint is NOT modified.
- Part122 checkpoints are NOT modified.
- No retraining occurs.
- Canonical MONAI SwinUNETR architecture is enforced by Part113.
- The existing Part9 / Part11 loader contract is reused unchanged.
- Validation cohort remains the established Part99 100-row cohort.

Part122 best checkpoint:
    outputs/segmentation/rsna_part122_controlled_background_calibrated_loss_retraining/
        checkpoints/best_part122_model.pth

Results:
    outputs/segmentation/rsna_part123_part122_best_full_validation_inference_and_distribution_audit/
        inference/
        audit/
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd


PROJECT_ROOT = (
    Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
)

SRC_DIR = PROJECT_ROOT / "src"

VAL_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part99_clinical_oriented_finetuning"
    / "part99_validation_cohort_reconstructed.csv"
)

PART120_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part122_controlled_background_calibrated_loss_retraining"
)

CHECKPOINT = PART120_DIR / "checkpoints" / "best_part122_model.pth"

PART123_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part123_part122_best_full_validation_inference_and_distribution_audit"
)

INFERENCE_DIR = PART123_DIR / "inference"
AUDIT_DIR = PART123_DIR / "audit"
REPORT_DIR = PROJECT_ROOT / "reports"

P113_PATH = SRC_DIR / "segmentation_rsna_part113_robust_loader_contract_and_full_inference.py"
# Part114 had a corrected/fixed revision during the earlier workflow.
# Resolve the actual Part114 script present in the user's src directory instead
# of assuming one exact filename. This prevents a false "MISSING" failure when
# the file has a slightly different suffix/name.
P114_CANDIDATES = [
    SRC_DIR / "segmentation_rsna_part114_prediction_distribution_spatial_quality_audit_fixed.py",
    SRC_DIR / "segmentation_rsna_part114_prediction_distribution_spatial_quality_audit.py",
]

def resolve_part114_script() -> Path:
    for candidate in P114_CANDIDATES:
        if candidate.exists():
            return candidate

    matches = sorted(SRC_DIR.glob("segmentation_rsna_part114*.py"))
    if matches:
        # Prefer a script whose name indicates the corrected/fixed version.
        fixed = [x for x in matches if "fixed" in x.name.lower()]
        return fixed[0] if fixed else matches[-1]

    # Return the canonical expected path so the caller can report a useful
    # missing-input message.
    return P114_CANDIDATES[0]

P114_PATH = resolve_part114_script()

INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 88)
    print(text)
    print("=" * 88)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_lineage_summary(
    checkpoint_sha: str,
    inference_module: Any,
    audit_module: Any,
    inference_rc: int,
    audit_rc: int,
    elapsed: float,
) -> Path:
    path = REPORT_DIR / "part123_part122_best_full_validation_inference_and_distribution_audit_summary.json"

    payload: Dict[str, Any] = {
        "part": 121,
        "purpose": "Full 100-row validation inference and prediction-distribution audit using Part122 best checkpoint.",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "part122_directory": str(PART120_DIR),
        "part104_preserved": True,
        "validation_cohort": str(VAL_COHORT),
        "validation_cohort_rows": int(len(pd.read_csv(VAL_COHORT))) if VAL_COHORT.exists() else None,
        "part113_reused": True,
        "part114_reused": True,
        "part113_return_code": int(inference_rc),
        "part114_return_code": int(audit_rc),
        "inference_output_directory": str(INFERENCE_DIR),
        "audit_output_directory": str(AUDIT_DIR),
        "elapsed_seconds": float(elapsed),
        "status": "PASS" if inference_rc == 0 and audit_rc == 0 else "FAIL",
        "note": "Segmentation Dice values in this pipeline are against existing development/pseudo-mask targets, not clinical expert ground truth.",
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    start_total = time.perf_counter()

    banner("PART 121 — PART120 BEST CHECKPOINT FULL VALIDATION + DISTRIBUTION AUDIT")

    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Python             : {sys.executable}")
    print(f"Validation cohort  : {VAL_COHORT}")
    print(f"Part122 checkpoint : {CHECKPOINT}")
    print(f"Part123 output     : {PART123_DIR}")

    banner("1. PART120 CHECKPOINT / COHORT LINEAGE VALIDATION")

    required = {
        "validation cohort": VAL_COHORT,
        "Part122 best checkpoint": CHECKPOINT,
        "Part113 script": P113_PATH,
        "Part114 audit script": P114_PATH,
    }

    for name, path in required.items():
        print(f"{name:<28}: {'FOUND' if path.exists() else 'MISSING'}")
        if not path.exists():
            print("\nFINAL STATUS: FAIL — REQUIRED_INPUT_MISSING")
            return 1

    cohort_df = pd.read_csv(VAL_COHORT)

    if len(cohort_df) != 100:
        print(f"Unexpected validation cohort rows: {len(cohort_df)}")
        print("\nFINAL STATUS: FAIL — VALIDATION_COHORT_ROW_COUNT_MISMATCH")
        return 1

    checkpoint_sha = sha256_file(CHECKPOINT)

    print(f"Validation rows     : {len(cohort_df)}")
    print(
        f"Unique studies      : "
        f"{cohort_df['study_id'].nunique() if 'study_id' in cohort_df.columns else 'N/A'}"
    )
    print(f"Part122 checkpoint  : FOUND")
    print(f"Part122 SHA256      : {checkpoint_sha}")
    print("Part104 modification: NO")

    banner("2. REUSE ESTABLISHED PART113 FULL-INFERENCE PIPELINE")

    try:
        p113 = import_from_path("part123_part113", P113_PATH)

        # Override only evaluation paths/checkpoint identity. The Part113
        # loader contract, canonical SwinUNETR construction, normalization,
        # prediction logic, and 100-row iteration remain unchanged.
        p113.PROJECT_ROOT = PROJECT_ROOT
        p113.SRC_DIR = SRC_DIR
        p113.VAL_COHORT = VAL_COHORT
        p113.CHECKPOINT = CHECKPOINT
        p113.EXPECTED_SHA256 = checkpoint_sha

        p113.OUTPUT_DIR = INFERENCE_DIR
        p113.PRED_DIR = INFERENCE_DIR / "predictions"
        p113.REPORT_DIR = REPORT_DIR

        p113.CASE_CSV = INFERENCE_DIR / "part123_validation_case_inference.csv"
        p113.SUMMARY_JSON = (
            REPORT_DIR / "part123_validation_inference_summary.json"
        )
        p113.REPORT_TXT = (
            REPORT_DIR / "part123_validation_inference_report.txt"
        )

        p113.PRED_DIR.mkdir(parents=True, exist_ok=True)

        inference_rc = int(p113.main())

    except Exception as exc:
        print(f"Part113 execution failed: {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — PART113_EXECUTION_FAILED")
        return 1

    gc.collect()

    if inference_rc != 0:
        print(f"Part113 return code: {inference_rc}")
        print("\nFINAL STATUS: FAIL — FULL_VALIDATION_INFERENCE_FAILED")
        return 1

    case_csv = INFERENCE_DIR / "part123_validation_case_inference.csv"
    pred_dir = INFERENCE_DIR / "predictions"

    if not case_csv.exists() or not pred_dir.exists():
        print("Part123 inference artifacts are missing.")
        print("\nFINAL STATUS: FAIL — INFERENCE_ARTIFACTS_MISSING")
        return 1

    prediction_count = len(list(pred_dir.glob("*.npz")))
    print(f"Part123 prediction files: {prediction_count}")

    if prediction_count != 100:
        print("\nFINAL STATUS: FAIL — PREDICTION_FILE_COUNT_MISMATCH")
        return 1

    banner("3. REUSE ESTABLISHED PART114 DISTRIBUTION / SPATIAL AUDIT")

    try:
        p114 = import_from_path("part123_part114", P114_PATH)

        # Point Part114 at the NEW Part123 inference outputs. Its audit
        # calculations remain unchanged.
        p114.PROJECT_ROOT = PROJECT_ROOT
        p114.INPUT_DIR = INFERENCE_DIR
        p114.CASE_CSV = case_csv
        p114.PRED_DIR = pred_dir
        p114.OUTPUT_DIR = AUDIT_DIR
        p114.REPORT_DIR = REPORT_DIR

        # Part114 writes its own standard filenames under OUTPUT_DIR/REPORT_DIR.
        audit_rc = int(p114.main())

    except Exception as exc:
        print(f"Part114 execution failed: {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — PART114_EXECUTION_FAILED")
        return 1

    elapsed = time.perf_counter() - start_total

    summary_path = write_lineage_summary(
        checkpoint_sha=checkpoint_sha,
        inference_module=p113,
        audit_module=p114,
        inference_rc=inference_rc,
        audit_rc=audit_rc,
        elapsed=elapsed,
    )

    banner("PART 121 FINAL RESULT")

    print(f"Part122 best checkpoint SHA : {checkpoint_sha}")
    print(f"Validation cohort rows      : {len(cohort_df)}")
    print(f"Prediction files             : {prediction_count}")
    print(f"Part113 inference return     : {inference_rc}")
    print(f"Part114 audit return         : {audit_rc}")
    print(f"Elapsed seconds              : {elapsed:.2f}")
    print(f"Lineage summary              : {summary_path}")

    if inference_rc == 0 and audit_rc == 0:
        print("\nFINAL STATUS: PASS — PART120 FULL VALIDATION INFERENCE + DISTRIBUTION AUDIT COMPLETED")
        print("Decision: DO NOT replace Part104 until Part123 audit results are reviewed.")
        print("Note: Part123 validation metrics use development/pseudo-mask targets.")
        return 0

    print("\nFINAL STATUS: FAIL — PART123_COMPLETED_WITH_FAILURE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
