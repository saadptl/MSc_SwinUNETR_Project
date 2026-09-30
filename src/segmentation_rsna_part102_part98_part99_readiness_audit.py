from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch


# =============================================================================
# PART 102
# Part 98 / Part 99 Segmentation Artifact Recovery & Clinical Training Readiness
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

SEG_ROOT = ROOT / "outputs" / "segmentation"

PART98_DIR = SEG_ROOT / "rsna_part98_strong_full_cohort_training"
PART99_DIR = SEG_ROOT / "rsna_part99_clinical_oriented_finetuning"

PART98_CKPT = PART98_DIR / "part98_best_model.pth"

PART15_INIT = (
    SEG_ROOT
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs" / "segmentation" / "part102_part98_part99_readiness_audit"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = REPORT_DIR / "part102_part98_part99_readiness_summary.json"
REPORT_TXT = REPORT_DIR / "part102_part98_part99_readiness_report.txt"

EXPECTED_PART15_SHA256 = (
    "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
)

EXPECTED_PARAMS = 4_078_116
EXPECTED_TRAIN_ROWS = 500
EXPECTED_VAL_ROWS = 100


# =============================================================================
# Utility functions
# =============================================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024.0 * 1024.0)


def print_check(label: str, status: str) -> None:
    print(f"{label:<62} {status}")


def safe_load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(checkpoint, dict):
            return checkpoint
        return {"_raw_checkpoint": checkpoint}
    except Exception as exc:
        print(f"Checkpoint loading error: {exc}")
        return None


def find_files(root: Path, patterns: List[str]) -> List[Path]:
    results: List[Path] = []

    if not root.exists():
        return results

    for pattern in patterns:
        results.extend(root.rglob(pattern))

    unique = sorted(set(results))
    return unique


def describe_csv(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "rows": None,
        "columns": [],
        "study_id_column": None,
        "unique_first_column": None,
    }

    if not path.exists():
        return info

    try:
        df = pd.read_csv(path)

        info["rows"] = int(len(df))
        info["columns"] = [str(c) for c in df.columns]

        for candidate in ["study_id", "case_id", "series_id"]:
            if candidate in df.columns:
                info["study_id_column"] = candidate
                info["unique_first_column"] = int(df[candidate].nunique())
                break

        if info["study_id_column"] is None and len(df.columns) > 0:
            first = df.columns[0]
            info["study_id_column"] = str(first)
            info["unique_first_column"] = int(df[first].nunique())

    except Exception as exc:
        info["error"] = str(exc)

    return info


def compare_csv_overlap(path_a: Path, path_b: Path) -> Dict[str, Any]:
    result = {
        "available": False,
        "column_a": None,
        "column_b": None,
        "rows_a": None,
        "rows_b": None,
        "overlap": None,
    }

    if not path_a.exists() or not path_b.exists():
        return result

    try:
        a = pd.read_csv(path_a)
        b = pd.read_csv(path_b)

        candidates = ["study_id", "case_id", "series_id"]

        col_a = next((c for c in candidates if c in a.columns), None)
        col_b = next((c for c in candidates if c in b.columns), None)

        if col_a is None and len(a.columns) > 0:
            col_a = a.columns[0]

        if col_b is None and len(b.columns) > 0:
            col_b = b.columns[0]

        if col_a is None or col_b is None:
            return result

        set_a = set(a[col_a].astype(str))
        set_b = set(b[col_b].astype(str))

        result["available"] = True
        result["column_a"] = str(col_a)
        result["column_b"] = str(col_b)
        result["rows_a"] = int(len(a))
        result["rows_b"] = int(len(b))
        result["overlap"] = int(len(set_a.intersection(set_b)))

    except Exception as exc:
        result["error"] = str(exc)

    return result


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print("=" * 82)
    print("PART 102 — PART 98 / PART 99 SEGMENTATION READINESS AUDIT")
    print("=" * 82)

    print(f"Project root : {ROOT}")
    print(f"Device       : {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    print()
    print("CHECKING PART 98 DIRECTORY")
    print("-" * 82)

    print_check(
        "Part98 directory",
        "PASS" if PART98_DIR.exists() else "FAIL",
    )

    if PART98_DIR.exists():
        print(f"Part98 directory : {PART98_DIR}")

    # -------------------------------------------------------------------------
    # Search for Part98-related artifacts
    # -------------------------------------------------------------------------

    print()
    print("SEARCHING FOR PART 98 ARTIFACTS")
    print("-" * 82)

    part98_files = find_files(
        SEG_ROOT,
        [
            "*part98*",
            "*Part98*",
        ],
    )

    if part98_files:
        for p in part98_files:
            print(f"FOUND : {p}")
    else:
        print("No Part98-related files found.")

    # -------------------------------------------------------------------------
    # Check expected artifacts
    # -------------------------------------------------------------------------

    expected_files = [
        PART98_CKPT,
        PART98_DIR / "part98_train_cohort.csv",
        PART98_DIR / "part98_validation_cohort.csv",
        PART98_DIR / "part98_training_history.csv",
        PART98_DIR / "part98_final_results.csv",
    ]

    print()
    print("EXPECTED PART 98 ARTIFACTS")
    print("-" * 82)

    expected_status = {}

    for path in expected_files:
        exists = path.exists()

        expected_status[str(path)] = {
            "exists": exists,
            "size_mb": file_size_mb(path),
        }

        print_check(
            path.name,
            "PASS" if exists else "MISSING",
        )

    # -------------------------------------------------------------------------
    # Search for cohort CSVs elsewhere
    # -------------------------------------------------------------------------

    print()
    print("SEARCHING FOR POSSIBLE RECOVERABLE COHORT FILES")
    print("-" * 82)

    cohort_candidates = find_files(
        SEG_ROOT,
        [
            "*train*cohort*.csv",
            "*validation*cohort*.csv",
            "*train*manifest*.csv",
            "*validation*manifest*.csv",
        ],
    )

    cohort_descriptions = []

    for p in cohort_candidates:
        info = describe_csv(p)
        cohort_descriptions.append(info)

        print(
            f"{p.name:<45} "
            f"rows={str(info.get('rows')):<6} "
            f"column={str(info.get('study_id_column'))}"
        )

    # -------------------------------------------------------------------------
    # Part15 initialization verification
    # -------------------------------------------------------------------------

    print()
    print("CHECKING PART 15 INITIALIZATION")
    print("-" * 82)

    part15_status = {
        "exists": PART15_INIT.exists(),
        "sha256": None,
        "expected_sha256": EXPECTED_PART15_SHA256,
        "hash_match": False,
    }

    if PART15_INIT.exists():
        actual_hash = sha256_file(PART15_INIT)
        part15_status["sha256"] = actual_hash
        part15_status["hash_match"] = actual_hash == EXPECTED_PART15_SHA256

        print_check(
            "Part15 initialization checkpoint",
            "PASS" if part15_status["hash_match"] else "FAIL",
        )

        print(f"SHA256 : {actual_hash}")
    else:
        print_check("Part15 initialization checkpoint", "MISSING")

    # -------------------------------------------------------------------------
    # Part98 checkpoint inspection
    # -------------------------------------------------------------------------

    print()
    print("INSPECTING PART 98 BEST CHECKPOINT")
    print("-" * 82)

    checkpoint_status: Dict[str, Any] = {
        "exists": PART98_CKPT.exists(),
        "size_mb": file_size_mb(PART98_CKPT),
        "parameter_count": None,
        "epoch": None,
        "keys": [],
        "model_state_dict_present": False,
        "optimizer_state_dict_present": False,
        "strict_architecture_possible": False,
    }

    checkpoint = safe_load_checkpoint(PART98_CKPT)

    if checkpoint is None:
        print_check("Part98 best checkpoint", "MISSING / UNREADABLE")
    else:
        checkpoint_status["keys"] = list(checkpoint.keys())

        checkpoint_status["model_state_dict_present"] = (
            "model_state_dict" in checkpoint
        )

        checkpoint_status["optimizer_state_dict_present"] = (
            "optimizer_state_dict" in checkpoint
        )

        checkpoint_status["epoch"] = checkpoint.get("epoch")

        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]

            # Parameter count cannot always be inferred from state_dict alone,
            # but we can estimate tensor element count.
            try:
                total_elements = 0
                for value in state.values():
                    if torch.is_tensor(value):
                        total_elements += value.numel()

                checkpoint_status["state_dict_tensor_elements"] = int(
                    total_elements
                )
            except Exception:
                pass

        print_check(
            "Part98 best checkpoint",
            "PASS",
        )

        print(f"Size MB : {checkpoint_status['size_mb']:.3f}")
        print(f"Epoch   : {checkpoint_status['epoch']}")
        print(f"Keys    : {checkpoint_status['keys']}")

    # -------------------------------------------------------------------------
    # Import model factory from Part11
    # -------------------------------------------------------------------------

    print()
    print("CHECKING SWIN-UNETR MODEL COMPATIBILITY")
    print("-" * 82)

    model_status: Dict[str, Any] = {
        "imported": False,
        "parameter_count": None,
        "architecture_match": False,
        "part98_strict_load": None,
        "error": None,
    }

    try:
        sys.path.insert(0, str(ROOT / "src"))

        import segmentation_rsna_part11_controlled_pilot_training as part11

        model = part11.create_model("cpu")

        model_status["imported"] = True
        model_status["parameter_count"] = sum(
            p.numel() for p in model.parameters()
        )

        model_status["architecture_match"] = (
            model_status["parameter_count"] == EXPECTED_PARAMS
        )

        print_check(
            "Part11 Swin-UNETR model factory",
            "PASS",
        )

        print(f"Parameters : {model_status['parameter_count']}")
        print(
            f"Expected   : {EXPECTED_PARAMS}"
        )

        if checkpoint is not None and "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=True,
                )

                model_status["part98_strict_load"] = True

                print_check(
                    "Part98 checkpoint strict load",
                    "PASS",
                )

            except Exception as exc:
                model_status["part98_strict_load"] = False
                model_status["error"] = str(exc)

                print_check(
                    "Part98 checkpoint strict load",
                    "FAIL",
                )

                print(f"Error : {exc}")

    except Exception as exc:
        model_status["error"] = str(exc)

        print_check(
            "Part11 Swin-UNETR model factory",
            "FAIL",
        )

        print(f"Error : {exc}")

    # -------------------------------------------------------------------------
    # Inspect all likely Part98 cohort candidates
    # -------------------------------------------------------------------------

    print()
    print("COHORT RECOVERY ANALYSIS")
    print("-" * 82)

    train_candidates = []
    val_candidates = []

    for info in cohort_descriptions:
        path = Path(info["path"])
        name = path.name.lower()

        if "train" in name:
            train_candidates.append(path)

        if "validation" in name or "val" in name:
            val_candidates.append(path)

    print(f"Train-like CSV candidates : {len(train_candidates)}")
    print(f"Val-like CSV candidates   : {len(val_candidates)}")

    # Prefer exact expected paths if present.
    expected_train = PART98_DIR / "part98_train_cohort.csv"
    expected_val = PART98_DIR / "part98_validation_cohort.csv"

    recovered_train = expected_train if expected_train.exists() else None
    recovered_val = expected_val if expected_val.exists() else None

    # If missing, look for likely expanded cohorts based on row counts.
    if recovered_train is None:
        for path in train_candidates:
            try:
                rows = len(pd.read_csv(path))
                if rows == EXPECTED_TRAIN_ROWS:
                    recovered_train = path
                    break
            except Exception:
                continue

    if recovered_val is None:
        for path in val_candidates:
            try:
                rows = len(pd.read_csv(path))
                if rows == EXPECTED_VAL_ROWS:
                    recovered_val = path
                    break
            except Exception:
                continue

    print()
    print(
        "Recovered/identified train cohort :",
        recovered_train if recovered_train else "NOT FOUND",
    )
    print(
        "Recovered/identified val cohort   :",
        recovered_val if recovered_val else "NOT FOUND",
    )

    train_info = (
        describe_csv(recovered_train)
        if recovered_train
        else {"exists": False}
    )

    val_info = (
        describe_csv(recovered_val)
        if recovered_val
        else {"exists": False}
    )

    # -------------------------------------------------------------------------
    # Overlap audit
    # -------------------------------------------------------------------------

    overlap_info = compare_csv_overlap(
        recovered_train,
        recovered_val,
    ) if recovered_train and recovered_val else {
        "available": False,
        "overlap": None,
    }

    print()
    print("TRAIN / VALIDATION OVERLAP")
    print("-" * 82)

    if overlap_info.get("available"):
        print(f"Train rows : {overlap_info['rows_a']}")
        print(f"Val rows   : {overlap_info['rows_b']}")
        print(f"Overlap    : {overlap_info['overlap']}")

        print_check(
            "Train/validation study overlap",
            "PASS" if overlap_info["overlap"] == 0 else "FAIL",
        )
    else:
        print_check(
            "Train/validation overlap audit",
            "NOT AVAILABLE",
        )

    # -------------------------------------------------------------------------
    # Check Part99 artifacts
    # -------------------------------------------------------------------------

    print()
    print("CHECKING PART 99 ARTIFACTS")
    print("-" * 82)

    part99_files = find_files(
        SEG_ROOT,
        [
            "*part99*",
            "*Part99*",
        ],
    )

    if part99_files:
        for p in part99_files:
            print(f"FOUND : {p}")
    else:
        print("No Part99 artifacts found.")

    # -------------------------------------------------------------------------
    # Final decision
    # -------------------------------------------------------------------------

    required_for_part99 = [
        PART98_CKPT.exists(),
        recovered_train is not None,
        recovered_val is not None,
        part15_status["hash_match"],
        model_status["imported"],
        model_status["architecture_match"],
        model_status["part98_strict_load"] is True,
        overlap_info.get("available") and overlap_info.get("overlap") == 0,
    ]

    if all(required_for_part99):
        final_status = "READY_FOR_PART99"

    elif PART98_CKPT.exists() and (
        recovered_train is not None or recovered_val is not None
    ):
        final_status = "RECOVERABLE_PART98_ARTIFACTS"

    else:
        final_status = "PART98_REQUIRES_RETRAINING"

    # -------------------------------------------------------------------------
    # Save recovered artifact index
    # -------------------------------------------------------------------------

    artifact_rows = []

    for p in part98_files:
        artifact_rows.append(
            {
                "artifact": p.name,
                "path": str(p),
                "exists": True,
                "size_mb": file_size_mb(p),
                "sha256": sha256_file(p)
                if p.is_file()
                else None,
            }
        )

    artifact_df = pd.DataFrame(artifact_rows)

    artifact_csv = OUTPUT_DIR / "part102_part98_artifact_inventory.csv"

    if not artifact_df.empty:
        artifact_df.to_csv(
            artifact_csv,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=["artifact", "path", "exists", "size_mb", "sha256"]
        ).to_csv(
            artifact_csv,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    summary = {
        "part": 102,
        "title": "Part 98 / Part 99 Segmentation Readiness Audit",
        "project_root": str(ROOT),
        "part98_directory": str(PART98_DIR),
        "part99_directory": str(PART99_DIR),
        "part98_expected_artifacts": expected_status,
        "part98_discovered_files": [str(p) for p in part98_files],
        "part98_cohort_candidates": cohort_descriptions,
        "recovered_train_cohort": str(recovered_train)
        if recovered_train
        else None,
        "recovered_validation_cohort": str(recovered_val)
        if recovered_val
        else None,
        "train_cohort_info": train_info,
        "validation_cohort_info": val_info,
        "train_validation_overlap": overlap_info,
        "part15_initialization": part15_status,
        "part98_checkpoint": checkpoint_status,
        "model_compatibility": model_status,
        "final_status": final_status,
        "do_not_retrain_without_audit": True,
        "expected_part98_train_rows": EXPECTED_TRAIN_ROWS,
        "expected_part98_validation_rows": EXPECTED_VAL_ROWS,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------------------

    lines = [
        "=" * 82,
        "PART 102 — PART 98 / PART 99 SEGMENTATION READINESS AUDIT",
        "=" * 82,
        "",
        f"Project root: {ROOT}",
        "",
        "PART 98 CHECKPOINT",
        "-" * 82,
        f"Exists: {checkpoint_status['exists']}",
        f"Size MB: {checkpoint_status['size_mb']:.3f}",
        f"Epoch: {checkpoint_status['epoch']}",
        "",
        "PART 15 INITIALIZATION",
        "-" * 82,
        f"Exists: {part15_status['exists']}",
        f"SHA256: {part15_status['sha256']}",
        f"Expected SHA256: {EXPECTED_PART15_SHA256}",
        f"Hash match: {part15_status['hash_match']}",
        "",
        "MODEL COMPATIBILITY",
        "-" * 82,
        f"Imported: {model_status['imported']}",
        f"Parameters: {model_status['parameter_count']}",
        f"Expected parameters: {EXPECTED_PARAMS}",
        f"Architecture match: {model_status['architecture_match']}",
        f"Part98 strict load: {model_status['part98_strict_load']}",
        "",
        "COHORT RECOVERY",
        "-" * 82,
        f"Train cohort: {recovered_train}",
        f"Validation cohort: {recovered_val}",
        f"Train rows: {train_info.get('rows')}",
        f"Validation rows: {val_info.get('rows')}",
        f"Train/validation overlap: {overlap_info.get('overlap')}",
        "",
        "FINAL DECISION",
        "-" * 82,
        final_status,
        "",
        "INTERPRETATION",
        "-" * 82,
    ]

    if final_status == "READY_FOR_PART99":
        lines.extend(
            [
                "Part98 artifacts are internally consistent enough to proceed.",
                "Part99 may be rerun using the identified Part98 checkpoint and cohorts.",
                "No additional recovery or Part98 retraining is required by this audit.",
            ]
        )

    elif final_status == "RECOVERABLE_PART98_ARTIFACTS":
        lines.extend(
            [
                "Some Part98 artifacts exist, but the expected Part99 inputs are",
                "not yet completely verified.",
                "Do not start another long training run until the missing artifact",
                "is reconstructed or its correct location is identified.",
            ]
        )

    else:
        lines.extend(
            [
                "The Part98 artifact set is insufficient for a trustworthy Part99 run.",
                "If Part98 training actually completed, inspect the terminal output",
                "and training directory before retraining.",
                "Do not claim a Part98 result without a valid checkpoint and cohort.",
            ]
        )

    lines.extend(
        [
            "",
            "IMPORTANT",
            "-" * 82,
            "This audit does not establish clinical validity.",
            "A stronger segmentation model still requires independent external",
            "validation, expert annotation quality, robustness testing, calibration,",
            "and prospective clinical evaluation before clinical deployment.",
            "",
            "=" * 82,
        ]
    )

    REPORT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal final result
    # -------------------------------------------------------------------------

    print()
    print("=" * 82)
    print("PART 102 FINAL RESULT")
    print("=" * 82)

    print(f"Part98 checkpoint available : {checkpoint_status['exists']}")
    print(
        "Part98 train cohort        :",
        "FOUND" if recovered_train else "MISSING",
    )
    print(
        "Part98 validation cohort  :",
        "FOUND" if recovered_val else "MISSING",
    )
    print(
        "Part15 initialization     :",
        "VERIFIED" if part15_status["hash_match"] else "NOT VERIFIED",
    )
    print(
        "Model compatibility       :",
        "PASS"
        if model_status["architecture_match"]
        and model_status["part98_strict_load"] is True
        else "FAIL",
    )
    print(
        "Train/val overlap         :",
        overlap_info.get("overlap")
        if overlap_info.get("available")
        else "NOT AVAILABLE",
    )

    print()
    print(f"FINAL STATUS: {final_status}")

    print()
    print("OUTPUTS:")
    print(artifact_csv)
    print(SUMMARY_JSON)
    print(REPORT_TXT)

    print("=" * 82)


if __name__ == "__main__":
    main()