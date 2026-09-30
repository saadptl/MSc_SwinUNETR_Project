from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "outputs" / "segmentation"

PART98_DIR = SEG / "rsna_part98_strong_full_cohort_training"
PART99_DIR = SEG / "rsna_part99_clinical_oriented_finetuning"

PART98_BEST = PART98_DIR / "checkpoints" / "part98_best_model.pth"
PART99_BEST = PART99_DIR / "checkpoints" / "part99_best_model.pth"
PART99_FINAL = PART99_DIR / "checkpoints" / "part99_final_model.pth"

PART98_HISTORY = PART98_DIR / "part98_training_history.csv"
PART99_HISTORY = PART99_DIR / "part99_training_history.csv"
PART99_CASE_METRICS = PART99_DIR / "part99_validation_case_metrics.csv"

OUT_DIR = SEG / "rsna_part104_final_checkpoint_selection"
REPORT_DIR = ROOT / "reports"

MODEL_FACTORY = ROOT / "src" / "model.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


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
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "size_mb": None,
        "sha256": None,
        "key_count": None,
        "parameter_count": None,
        "epoch": None,
    }

    if not path.exists():
        return result

    result["size_mb"] = round(path.stat().st_size / (1024 ** 2), 3)
    result["sha256"] = sha256(path)

    try:
        obj = load_checkpoint(path)
        state = extract_state_dict(obj)

        if state is None:
            result["error"] = "No recognizable state_dict found."
            return result

        result["readable"] = True
        result["key_count"] = len(state)
        result["parameter_count"] = sum(
            v.numel() for v in state.values() if torch.is_tensor(v)
        )

        if isinstance(obj, dict):
            for key in ("epoch", "current_epoch"):
                if key in obj:
                    try:
                        result["epoch"] = int(obj[key])
                    except Exception:
                        pass

    except Exception as exc:
        result["error"] = repr(exc)

    return result


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_metric(row: dict, candidates: list[str]):
    lowered = {str(k).lower(): k for k in row.keys()}

    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass

    return None


def history_summary(path: Path) -> dict:
    rows = read_csv(path)

    summary = {
        "exists": path.exists(),
        "rows": len(rows),
        "best_epoch": None,
        "best_metric": None,
        "metric_name": None,
        "latest_epoch": None,
        "latest_metric": None,
    }

    if not rows:
        return summary

    epoch_rows = []
    for row in rows:
        epoch = find_metric(row, ["epoch"])
        if epoch is None:
            continue

        metric = None
        metric_name = None

        for name in [
            "val_global_fg_dice",
            "global_fg_dice",
            "val_fg_dice",
            "val_mean_case_dice",
            "mean_case_dice",
            "val_dice",
            "dice",
        ]:
            value = find_metric(row, [name])
            if value is not None:
                metric = value
                metric_name = name
                break

        if metric is not None:
            epoch_rows.append((int(epoch), metric, metric_name))

    if not epoch_rows:
        return summary

    best = max(epoch_rows, key=lambda x: x[1])
    latest = max(epoch_rows, key=lambda x: x[0])

    summary["best_epoch"] = best[0]
    summary["best_metric"] = best[1]
    summary["metric_name"] = best[2]
    summary["latest_epoch"] = latest[0]
    summary["latest_metric"] = latest[1]

    return summary


def case_metric_summary(path: Path) -> dict:
    rows = read_csv(path)

    if not rows:
        return {
            "exists": path.exists(),
            "rows": len(rows),
            "columns": [],
            "numeric_summary": {},
        }

    numeric_columns = {}
    for key in rows[0].keys():
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                pass

        if values:
            numeric_columns[key] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

    return {
        "exists": True,
        "rows": len(rows),
        "columns": list(rows[0].keys()),
        "numeric_summary": numeric_columns,
    }


def compare_histories(h98: dict, h99: dict) -> dict:
    result = {}

    if h98["best_metric"] is not None and h99["best_metric"] is not None:
        result["best_metric_delta"] = h99["best_metric"] - h98["best_metric"]
        result["best_metric_relative_percent"] = (
            (h99["best_metric"] - h98["best_metric"])
            / h98["best_metric"]
            * 100.0
            if h98["best_metric"] != 0
            else None
        )

    if h98["latest_metric"] is not None and h99["latest_metric"] is not None:
        result["latest_metric_delta"] = h99["latest_metric"] - h98["latest_metric"]

    return result


def architecture_compare(a_path: Path, b_path: Path) -> dict:
    if not a_path.exists() or not b_path.exists():
        return {
            "available": False,
            "compatible": None,
            "reason": "One or both checkpoints are missing.",
        }

    try:
        a = extract_state_dict(load_checkpoint(a_path))
        b = extract_state_dict(load_checkpoint(b_path))

        if a is None or b is None:
            return {
                "available": True,
                "compatible": False,
                "reason": "Unable to extract state dictionaries.",
            }

        a_keys = set(a.keys())
        b_keys = set(b.keys())

        same_keys = a_keys == b_keys
        same_shapes = (
            same_keys
            and all(a[k].shape == b[k].shape for k in a_keys)
        )

        return {
            "available": True,
            "compatible": bool(same_keys and same_shapes),
            "same_keys": same_keys,
            "same_shapes": same_shapes,
            "a_keys": len(a_keys),
            "b_keys": len(b_keys),
        }
    except Exception as exc:
        return {
            "available": True,
            "compatible": False,
            "reason": repr(exc),
        }


def main():
    print("=" * 88)
    print("PART 104 — FINAL SEGMENTATION CHECKPOINT SELECTION + COMPARATIVE EVALUATION")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCHECKPOINT AVAILABILITY")
    print("-" * 88)

    p98 = inspect_checkpoint(PART98_BEST)
    p99 = inspect_checkpoint(PART99_BEST)
    p99_final = inspect_checkpoint(PART99_FINAL)

    for label, item in [
        ("Part98 best", p98),
        ("Part99 best", p99),
        ("Part99 final", p99_final),
    ]:
        state = "PASS" if item["exists"] and item["readable"] else "FAIL"
        print(f"{label:<24} {state}")
        if item["exists"]:
            print(f"  Size       : {item['size_mb']} MB")
            print(f"  SHA256     : {item['sha256']}")
            print(f"  Parameters : {item['parameter_count']}")

    print("\nTRAINING HISTORY COMPARISON")
    print("-" * 88)

    h98 = history_summary(PART98_HISTORY)
    h99 = history_summary(PART99_HISTORY)

    print(f"Part98 history rows : {h98['rows']}")
    print(f"Part98 best epoch  : {h98['best_epoch']}")
    print(f"Part98 best metric : {h98['best_metric']}")
    print(f"Part99 history rows : {h99['rows']}")
    print(f"Part99 best epoch  : {h99['best_epoch']}")
    print(f"Part99 best metric : {h99['best_metric']}")

    comparison = compare_histories(h98, h99)

    if comparison.get("best_metric_delta") is not None:
        print(
            f"Best-metric delta (Part99 - Part98) : "
            f"{comparison['best_metric_delta']:.6f}"
        )
        if comparison.get("best_metric_relative_percent") is not None:
            print(
                f"Relative change                     : "
                f"{comparison['best_metric_relative_percent']:.2f}%"
            )

    print("\nARCHITECTURE / LINEAGE CHECK")
    print("-" * 88)

    compatibility = architecture_compare(PART98_BEST, PART99_BEST)
    print(
        "Part98 best ↔ Part99 best : "
        + ("PASS" if compatibility.get("compatible") else "FAIL")
    )

    print("\nPART99 VALIDATION CASE METRICS")
    print("-" * 88)

    case_summary = case_metric_summary(PART99_CASE_METRICS)
    print(f"Rows : {case_summary['rows']}")

    for key, info in case_summary["numeric_summary"].items():
        if "dice" in key.lower():
            print(
                f"{key:<32} mean={info['mean']:.6f} "
                f"min={info['min']:.6f} max={info['max']:.6f}"
            )

    # Decision rule:
    # Prefer the Part99 best checkpoint only if its recorded best validation
    # metric is at least as good as Part98's. Otherwise retain Part98.
    #
    # This is a checkpoint-selection audit, not a clinical efficacy claim.
    if (
        p98["readable"]
        and p99["readable"]
        and h98["best_metric"] is not None
        and h99["best_metric"] is not None
    ):
        if h99["best_metric"] >= h98["best_metric"]:
            selected = PART99_BEST
            selected_label = "Part99 best"
            rationale = (
                "Part99 recorded best validation metric is greater than or equal "
                "to Part98 recorded best validation metric."
            )
        else:
            selected = PART98_BEST
            selected_label = "Part98 best"
            rationale = (
                "Part98 recorded best validation metric exceeds Part99's recorded "
                "best validation metric; Part98 is retained as the stronger "
                "recorded segmentation checkpoint."
            )
    elif p99["readable"]:
        selected = PART99_BEST
        selected_label = "Part99 best"
        rationale = (
            "Part99 best checkpoint is readable, but a direct metric comparison "
            "could not be completed from the available histories."
        )
    elif p98["readable"]:
        selected = PART98_BEST
        selected_label = "Part98 best"
        rationale = "Part99 best is unavailable; Part98 best is retained."
    else:
        selected = None
        selected_label = None
        rationale = "No usable final segmentation checkpoint is available."

    print("\nFINAL CHECKPOINT DECISION")
    print("-" * 88)

    if selected is not None:
        print(f"Selected checkpoint : {selected_label}")
        print(f"Path                : {selected}")
        print(f"Rationale           : {rationale}")

        final_dir = OUT_DIR / "checkpoints"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "final_segmentation_model.pth"

        # Copy rather than mutate the source checkpoint.
        import shutil
        shutil.copy2(selected, final_path)

        print(f"Published checkpoint: {final_path}")
    else:
        final_path = None
        print("NO FINAL CHECKPOINT PUBLISHED")

    result = {
        "part": 104,
        "status": (
            "PASS — FINAL_SEGMENTATION_CHECKPOINT_SELECTED"
            if final_path is not None
            else "FAIL — NO_USABLE_SEGMENTATION_CHECKPOINT"
        ),
        "selected_checkpoint": selected_label,
        "selected_source": str(selected) if selected else None,
        "published_checkpoint": str(final_path) if final_path else None,
        "rationale": rationale,
        "part98": p98,
        "part99_best": p99,
        "part99_final": p99_final,
        "part98_history": h98,
        "part99_history": h99,
        "comparison": comparison,
        "architecture_compatibility": compatibility,
        "part99_case_metrics": case_summary,
        "clinical_interpretation_note": (
            "These checkpoint-selection results are model-development metrics "
            "and must not be presented as clinical diagnostic accuracy or "
            "manual-ground-truth segmentation performance unless independently "
            "validated against appropriate expert annotations."
        ),
    }

    inventory_path = OUT_DIR / "part104_checkpoint_comparison.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "checkpoint",
            "exists",
            "readable",
            "size_mb",
            "sha256",
            "parameter_count",
            "best_epoch",
            "best_validation_metric",
            "selected",
        ])

        writer.writerow([
            "Part98 best",
            p98["exists"],
            p98["readable"],
            p98["size_mb"],
            p98["sha256"],
            p98["parameter_count"],
            h98["best_epoch"],
            h98["best_metric"],
            selected_label == "Part98 best",
        ])

        writer.writerow([
            "Part99 best",
            p99["exists"],
            p99["readable"],
            p99["size_mb"],
            p99["sha256"],
            p99["parameter_count"],
            h99["best_epoch"],
            h99["best_metric"],
            selected_label == "Part99 best",
        ])

        writer.writerow([
            "Part99 final",
            p99_final["exists"],
            p99_final["readable"],
            p99_final["size_mb"],
            p99_final["sha256"],
            p99_final["parameter_count"],
            h99["latest_epoch"],
            h99["latest_metric"],
            False,
        ])

    summary_path = REPORT_DIR / "part104_final_checkpoint_selection_summary.json"
    report_path = REPORT_DIR / "part104_final_checkpoint_selection_report.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    report = [
        "=" * 88,
        "PART 104 — FINAL SEGMENTATION CHECKPOINT SELECTION",
        "=" * 88,
        "",
        f"Project root: {ROOT}",
        f"Status: {result['status']}",
        "",
        "SELECTED CHECKPOINT",
        f"  Label: {selected_label}",
        f"  Source: {selected}",
        f"  Published: {final_path}",
        f"  Rationale: {rationale}",
        "",
        "PART98",
        f"  Best epoch: {h98['best_epoch']}",
        f"  Best validation metric: {h98['best_metric']}",
        f"  SHA256: {p98['sha256']}",
        "",
        "PART99",
        f"  Best epoch: {h99['best_epoch']}",
        f"  Best validation metric: {h99['best_metric']}",
        f"  Best SHA256: {p99['sha256']}",
        f"  Final SHA256: {p99_final['sha256']}",
        "",
        "COMPARISON",
        json.dumps(comparison, indent=2),
        "",
        "VALIDATION CASE METRICS",
        json.dumps(case_summary, indent=2),
        "",
        "IMPORTANT INTERPRETATION",
        result["clinical_interpretation_note"],
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PART 104 FINAL RESULT")
    print("=" * 88)
    print(result["status"])
    print("\nOUTPUTS:")
    print(inventory_path)
    print(summary_path)
    print(report_path)
    if final_path:
        print(final_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
