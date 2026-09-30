from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "outputs" / "segmentation"

FINAL_DIR = SEG / "rsna_part104_final_checkpoint_selection"
FINAL_CKPT = FINAL_DIR / "checkpoints" / "final_segmentation_model.pth"

PART98_DIR = SEG / "rsna_part98_strong_full_cohort_training"
PART98_VAL = PART98_DIR / "part98_validation_cohort.csv"
PART98_HISTORY = PART98_DIR / "part98_training_history.csv"

PART99_DIR = SEG / "rsna_part99_clinical_oriented_finetuning"
PART99_VAL = PART99_DIR / "part99_validation_cohort.csv"
PART99_CASE_METRICS = PART99_DIR / "part99_validation_case_metrics.csv"

OUT_DIR = SEG / "rsna_part105_final_inference_readiness"
REPORT_DIR = ROOT / "reports"

EXPECTED_PARAMETERS = 4_078_116


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "network", "net"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    return None


def inspect_checkpoint(path: Path) -> dict:
    result = {
        "exists": path.exists(),
        "readable": False,
        "sha256": None,
        "size_mb": None,
        "state_dict_keys": None,
        "tensor_parameter_count": None,
        "checkpoint_metadata": {},
        "errors": [],
    }

    if not path.exists():
        result["errors"].append("Final checkpoint does not exist.")
        return result

    result["size_mb"] = round(path.stat().st_size / (1024 ** 2), 3)
    result["sha256"] = sha256(path)

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        state = extract_state_dict(obj)

        if state is None:
            result["errors"].append("No recognizable state_dict found.")
            return result

        result["readable"] = True
        result["state_dict_keys"] = len(state)
        result["tensor_parameter_count"] = sum(
            value.numel()
            for value in state.values()
            if torch.is_tensor(value)
        )

        if isinstance(obj, dict):
            for key in (
                "epoch",
                "current_epoch",
                "best_epoch",
                "val_dice",
                "best_val_dice",
            ):
                if key in obj:
                    value = obj[key]
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        result["checkpoint_metadata"][key] = value

    except Exception as exc:
        result["errors"].append(repr(exc))

    return result


def cohort_audit(path: Path) -> dict:
    rows = read_csv(path)

    result = {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "unique_study_ids": None,
        "duplicate_study_ids": None,
    }

    if not rows:
        return result

    key = "study_id" if "study_id" in rows[0] else list(rows[0].keys())[0]
    ids = [
        str(row.get(key, "")).strip()
        for row in rows
        if str(row.get(key, "")).strip()
    ]

    result["id_column"] = key
    result["unique_study_ids"] = len(set(ids))
    result["duplicate_study_ids"] = len(ids) - len(set(ids))

    return result


def overlap_audit(train_path: Path, val_path: Path) -> dict:
    train_rows = read_csv(train_path)
    val_rows = read_csv(val_path)

    def get_ids(rows):
        if not rows:
            return set()
        key = "study_id" if "study_id" in rows[0] else list(rows[0].keys())[0]
        return {
            str(row.get(key, "")).strip()
            for row in rows
            if str(row.get(key, "")).strip()
        }

    train_ids = get_ids(train_rows)
    val_ids = get_ids(val_rows)
    overlap = train_ids & val_ids

    return {
        "train_unique_ids": len(train_ids),
        "validation_unique_ids": len(val_ids),
        "overlap": len(overlap),
        "pass": len(overlap) == 0,
    }


def main():
    print("=" * 88)
    print("PART 105 — FINAL SEGMENTATION INFERENCE READINESS AUDIT")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device  : {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCHECKING FINAL CHECKPOINT")
    print("-" * 88)

    checkpoint = inspect_checkpoint(FINAL_CKPT)

    if checkpoint["exists"]:
        print("final_segmentation_model.pth                 FOUND")
        print(f"Size MB                                      : {checkpoint['size_mb']}")
        print(f"SHA256                                       : {checkpoint['sha256']}")
        print(f"State-dict keys                              : {checkpoint['state_dict_keys']}")
        print(
            "Tensor parameter count                       : "
            f"{checkpoint['tensor_parameter_count']}"
        )
    else:
        print("final_segmentation_model.pth                 MISSING")

    parameter_pass = (
        checkpoint["readable"]
        and checkpoint["tensor_parameter_count"] == EXPECTED_PARAMETERS
    )

    print(
        "Parameter-count compatibility                : "
        + ("PASS" if parameter_pass else "FAIL")
    )

    print("\nCHECKING CANONICAL VALIDATION COHORT")
    print("-" * 88)

    val = cohort_audit(PART98_VAL)

    print(f"Path                                         : {PART98_VAL}")
    print(f"Exists                                       : {val['exists']}")
    print(f"Rows                                         : {val['rows']}")
    print(f"Unique study IDs                             : {val['unique_study_ids']}")
    print(f"Duplicate study IDs                          : {val['duplicate_study_ids']}")

    val_pass = (
        val["exists"]
        and val["rows"] == 100
        and val["unique_study_ids"] == 100
        and val["duplicate_study_ids"] == 0
    )

    print(
        "Canonical 100-study validation cohort       : "
        + ("PASS" if val_pass else "FAIL")
    )

    print("\nCHECKING TRAIN / VALIDATION ISOLATION")
    print("-" * 88)

    train_path = PART98_DIR / "part98_train_cohort.csv"
    split = overlap_audit(train_path, PART98_VAL)

    print(f"Train unique IDs                             : {split['train_unique_ids']}")
    print(f"Validation unique IDs                        : {split['validation_unique_ids']}")
    print(f"Overlap                                      : {split['overlap']}")
    print(
        "Train/validation isolation                   : "
        + ("PASS" if split["pass"] else "FAIL")
    )

    print("\nCHECKING PART104 SELECTION RECORD")
    print("-" * 88)

    comparison_path = FINAL_DIR / "part104_checkpoint_comparison.csv"
    comparison_rows = read_csv(comparison_path)

    selected_rows = [
        row for row in comparison_rows
        if str(row.get("selected", "")).strip().lower() in {"true", "1", "yes"}
    ]

    print(f"Part104 comparison file                     : {'FOUND' if comparison_rows else 'MISSING'}")
    print(f"Selected checkpoint records                 : {len(selected_rows)}")

    selection_pass = len(selected_rows) == 1
    if selection_pass:
        print(f"Selected source recorded                   : {selected_rows[0].get('checkpoint')}")

    print("\nCHECKING PART99 EXPERIMENT PRESERVATION")
    print("-" * 88)

    p99_best = PART99_DIR / "checkpoints" / "part99_best_model.pth"
    p99_final = PART99_DIR / "checkpoints" / "part99_final_model.pth"

    print(f"Part99 best checkpoint preserved             : {'PASS' if p99_best.exists() else 'FAIL'}")
    print(f"Part99 final checkpoint preserved            : {'PASS' if p99_final.exists() else 'FAIL'}")
    print(f"Part99 validation case metrics preserved    : {'PASS' if PART99_CASE_METRICS.exists() else 'FAIL'}")

    preservation_pass = (
        p99_best.exists()
        and p99_final.exists()
        and PART99_CASE_METRICS.exists()
    )

    print("\nCHECKING CPU CHECKPOINT RELOAD")
    print("-" * 88)

    reload_pass = False

    if checkpoint["readable"]:
        try:
            obj = torch.load(
                FINAL_CKPT,
                map_location="cpu",
                weights_only=False,
            )
            state = extract_state_dict(obj)
            reload_pass = state is not None
            print("CPU reload                                    : PASS")
        except Exception as exc:
            print("CPU reload                                    : FAIL")
            print(f"Error                                        : {exc}")
    else:
        print("CPU reload                                    : SKIPPED")

    print("\nCHECKING CUDA MODEL AVAILABILITY")
    print("-" * 88)

    cuda_pass = False
    cuda_error = None

    if torch.cuda.is_available() and checkpoint["readable"]:
        try:
            # Move the state tensors to CUDA as an inference-readiness smoke test.
            # This intentionally does not instantiate a new model or run training.
            obj = torch.load(
                FINAL_CKPT,
                map_location="cuda:0",
                weights_only=False,
            )
            state = extract_state_dict(obj)
            cuda_pass = state is not None
            print("CUDA checkpoint load                          : PASS")
        except Exception as exc:
            cuda_error = repr(exc)
            print("CUDA checkpoint load                          : FAIL")
            print(f"Error                                        : {exc}")
    elif not torch.cuda.is_available():
        cuda_error = "CUDA is not available."
        print("CUDA checkpoint load                          : SKIPPED — CUDA unavailable")
    else:
        cuda_error = "Checkpoint is unreadable."
        print("CUDA checkpoint load                          : SKIPPED")

    print("\nFINAL INFERENCE READINESS DECISION")
    print("-" * 88)

    overall_pass = (
        checkpoint["readable"]
        and parameter_pass
        and val_pass
        and split["pass"]
        and selection_pass
        and preservation_pass
        and reload_pass
        and cuda_pass
    )

    status = (
        "PASS — FINAL_SEGMENTATION_MODEL_INFERENCE_READY"
        if overall_pass
        else "FAIL — FINAL_SEGMENTATION_MODEL_REQUIRES_REVIEW"
    )

    print(f"Status                                       : {status}")

    result = {
        "part": 105,
        "status": status,
        "final_checkpoint": checkpoint,
        "expected_parameter_count": EXPECTED_PARAMETERS,
        "parameter_count_pass": parameter_pass,
        "canonical_validation_cohort": val,
        "train_validation_split": split,
        "part104_selection_records": selected_rows,
        "part104_comparison_file": str(comparison_path),
        "part99_preservation": {
            "best_exists": p99_best.exists(),
            "final_exists": p99_final.exists(),
            "case_metrics_exists": PART99_CASE_METRICS.exists(),
        },
        "cpu_reload_pass": reload_pass,
        "cuda_reload_pass": cuda_pass,
        "cuda_reload_error": cuda_error,
        "inference_scope": (
            "This is an inference-readiness audit only. It does not claim "
            "clinical diagnostic accuracy and does not replace evaluation "
            "against expert-annotated ground truth."
        ),
    }

    inventory_path = OUT_DIR / "part105_inference_readiness_inventory.csv"

    with inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "result", "detail"])
        writer.writerow([
            "final checkpoint exists/readable",
            checkpoint["readable"],
            str(FINAL_CKPT),
        ])
        writer.writerow([
            "parameter count",
            parameter_pass,
            checkpoint["tensor_parameter_count"],
        ])
        writer.writerow([
            "100-study validation cohort",
            val_pass,
            val["rows"],
        ])
        writer.writerow([
            "train/validation overlap",
            split["pass"],
            split["overlap"],
        ])
        writer.writerow([
            "Part104 selection record",
            selection_pass,
            len(selected_rows),
        ])
        writer.writerow([
            "Part99 experiment preserved",
            preservation_pass,
            "best + final + case metrics",
        ])
        writer.writerow([
            "CPU reload",
            reload_pass,
            "",
        ])
        writer.writerow([
            "CUDA reload",
            cuda_pass,
            cuda_error or "",
        ])

    summary_path = REPORT_DIR / "part105_final_inference_readiness_summary.json"
    report_path = REPORT_DIR / "part105_final_inference_readiness_report.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    report = [
        "=" * 88,
        "PART 105 — FINAL SEGMENTATION INFERENCE READINESS AUDIT",
        "=" * 88,
        "",
        f"Project root: {ROOT}",
        f"Status: {status}",
        "",
        "FINAL CHECKPOINT",
        f"  Path: {FINAL_CKPT}",
        f"  SHA256: {checkpoint['sha256']}",
        f"  Readable: {checkpoint['readable']}",
        f"  Parameter count: {checkpoint['tensor_parameter_count']}",
        "",
        "VALIDATION COHORT",
        f"  Rows: {val['rows']}",
        f"  Unique studies: {val['unique_study_ids']}",
        f"  Duplicates: {val['duplicate_study_ids']}",
        f"  Train/validation overlap: {split['overlap']}",
        "",
        "PART104 SELECTION",
        f"  Selected records: {len(selected_rows)}",
        f"  Selected source: {selected_rows[0].get('checkpoint') if selected_rows else 'NONE'}",
        "",
        "RUNTIME",
        f"  CPU reload: {reload_pass}",
        f"  CUDA reload: {cuda_pass}",
        "",
        "INTERPRETATION",
        result["inference_scope"],
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PART 105 OUTPUTS")
    print("=" * 88)
    print(inventory_path)
    print(summary_path)
    print(report_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
