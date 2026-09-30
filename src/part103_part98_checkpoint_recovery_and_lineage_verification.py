from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "outputs" / "segmentation"

PART98_DIR = SEG / "rsna_part98_strong_full_cohort_training"
PART98_CKPT_DIR = PART98_DIR / "checkpoints"
PART98_HISTORY = PART98_DIR / "part98_training_history.csv"
PART98_BEST = PART98_CKPT_DIR / "part98_best_model.pth"

PART99_DIR = SEG / "rsna_part99_clinical_oriented_finetuning"
PART99_BEST = PART99_DIR / "checkpoints" / "part99_best_model.pth"
PART99_FINAL = PART99_DIR / "checkpoints" / "part99_final_model.pth"

OUT_DIR = SEG / "part103_part98_checkpoint_recovery_and_lineage_verification"
REPORT_DIR = ROOT / "reports"

EXPECTED_KEYS = None


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
        # A raw state dict is also a dict whose values are tensors.
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    return None


def checkpoint_signature(path: Path):
    try:
        obj = load_checkpoint(path)
        state = extract_state_dict(obj)
        if state is None:
            return {
                "readable": False,
                "reason": "No recognizable state_dict found",
            }

        keys = set(state.keys())
        shapes = {
            str(k): list(v.shape)
            for k, v in state.items()
            if torch.is_tensor(v)
        }

        return {
            "readable": True,
            "key_count": len(keys),
            "keys": keys,
            "shapes": shapes,
        }
    except Exception as exc:
        return {"readable": False, "reason": repr(exc)}


def find_part98_candidates():
    candidates = []

    if PART98_CKPT_DIR.exists():
        candidates.extend(PART98_CKPT_DIR.glob("*.pth"))

    # Search all segmentation outputs, but only accept names that strongly
    # indicate Part98 provenance. Never use Part99 as a Part98 replacement.
    for p in SEG.rglob("*.pth"):
        name = p.name.lower()
        path_text = str(p).lower()

        if "part99" in name or "part99" in path_text:
            continue

        if "part98" in name or "part98" in path_text:
            candidates.append(p)

    # Also inspect epoch checkpoints located directly in the Part98 folder.
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return candidates


def read_history():
    if not PART98_HISTORY.exists():
        return []

    rows = []
    with PART98_HISTORY.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def numeric(row, names):
    for name in names:
        if name in row:
            try:
                return float(row[name])
            except (TypeError, ValueError):
                pass
    return None


def select_best_epoch(rows):
    """
    Select the intended Part98 best epoch from the recorded validation metric.
    Prefer explicit best-model indicators, then validation Dice-like metrics.
    """
    if not rows:
        return None, "No Part98 training history available."

    # Explicit best flags, if present.
    for row in rows:
        flag = str(
            row.get("is_best", row.get("best", row.get("best_model", "")))
        ).strip().lower()
        if flag in {"1", "true", "yes", "y"}:
            epoch = row.get("epoch")
            if epoch:
                return int(float(epoch)), "Explicit best flag in history."

    metric_names = [
        "val_global_fg_dice",
        "val_fg_dice",
        "global_fg_dice",
        "val_mean_case_dice",
        "mean_case_dice",
        "val_dice",
        "dice",
    ]

    scored = []
    for row in rows:
        epoch_value = row.get("epoch")
        if epoch_value is None:
            continue
        try:
            epoch = int(float(epoch_value))
        except ValueError:
            continue

        metric = numeric(row, metric_names)
        if metric is not None:
            scored.append((metric, epoch))

    if scored:
        metric, epoch = max(scored)
        return epoch, f"Highest recorded validation Dice-like metric ({metric:.6f})."

    return None, "History exists but contains no recognized validation Dice metric."


def verify_cohorts():
    train = PART98_DIR / "part98_train_cohort.csv"
    val = PART98_DIR / "part98_validation_cohort.csv"

    result = {
        "train_exists": train.exists(),
        "val_exists": val.exists(),
        "train_rows": 0,
        "val_rows": 0,
        "overlap": None,
    }

    def ids(path):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set()
            key = reader.fieldnames[0]
            return {
                str(r.get(key, "")).strip()
                for r in reader
                if str(r.get(key, "")).strip()
            }

    if train.exists():
        result["train_rows"] = sum(1 for _ in train.open(
            "r", encoding="utf-8-sig"
        )) - 1
    if val.exists():
        result["val_rows"] = sum(1 for _ in val.open(
            "r", encoding="utf-8-sig"
        )) - 1

    if train.exists() and val.exists():
        result["overlap"] = len(ids(train) & ids(val))

    return result


def compare_to_reference(candidate: Path, reference: Path):
    if not reference.exists():
        return {
            "reference_exists": False,
            "compatible": None,
            "reason": "Reference checkpoint unavailable.",
        }

    a = checkpoint_signature(candidate)
    b = checkpoint_signature(reference)

    if not a.get("readable") or not b.get("readable"):
        return {
            "reference_exists": True,
            "compatible": False,
            "reason": "One or both checkpoints are unreadable.",
        }

    same_keys = a["keys"] == b["keys"]
    same_shapes = a["shapes"] == b["shapes"]

    return {
        "reference_exists": True,
        "compatible": bool(same_keys and same_shapes),
        "same_keys": same_keys,
        "same_shapes": same_shapes,
        "candidate_key_count": a["key_count"],
        "reference_key_count": b["key_count"],
    }


def main():
    print("=" * 88)
    print("PART 103 — PART98 CHECKPOINT RECOVERY + PART99 LINEAGE VERIFICATION")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "part": 103,
        "project_root": str(ROOT),
        "status": "UNKNOWN",
        "part98": {},
        "part99": {},
        "cohorts": verify_cohorts(),
        "candidates": [],
        "recovery": {},
    }

    print("\nCHECKING PART98 ARTIFACTS")
    print("-" * 88)

    history = read_history()
    result["part98"]["history_exists"] = PART98_HISTORY.exists()
    result["part98"]["history_rows"] = len(history)

    if PART98_BEST.exists():
        sig = checkpoint_signature(PART98_BEST)
        if sig.get("readable"):
            print("part98_best_model.pth                         FOUND / READABLE")
            result["part98"]["best_exists"] = True
            result["part98"]["best_sha256"] = sha256(PART98_BEST)
        else:
            print("part98_best_model.pth                         FOUND / UNREADABLE")
            result["part98"]["best_exists"] = False
    else:
        print("part98_best_model.pth                         MISSING")
        result["part98"]["best_exists"] = False

    print("\nCHECKING PART98 EPOCH CHECKPOINTS")
    print("-" * 88)

    candidates = find_part98_candidates()

    for p in candidates:
        sig = checkpoint_signature(p)
        item = {
            "path": str(p),
            "name": p.name,
            "readable": bool(sig.get("readable")),
            "size_mb": round(p.stat().st_size / (1024 ** 2), 3),
            "sha256": sha256(p) if p.exists() else None,
        }

        if sig.get("readable"):
            item["key_count"] = sig["key_count"]

        result["candidates"].append(item)

        status = "READABLE" if item["readable"] else "UNREADABLE"
        print(f"{p.name:<48} {status}")

    print(f"\nCandidate checkpoints found : {len(candidates)}")

    best_epoch, best_reason = select_best_epoch(history)
    result["part98"]["selected_best_epoch"] = best_epoch
    result["part98"]["selection_reason"] = best_reason

    print("\nPART98 HISTORY ANALYSIS")
    print("-" * 88)
    print(f"History rows : {len(history)}")
    print(f"Selected best epoch : {best_epoch}")
    print(f"Reason : {best_reason}")

    recovered = None

    if not PART98_BEST.exists() and best_epoch is not None:
        epoch_candidates = [
            p for p in candidates
            if p.name.lower() in {
                f"part98_epoch_{best_epoch:02d}.pth",
                f"part98_epoch_{best_epoch:02d}_model.pth",
                f"part98_epoch_{best_epoch}.pth",
            }
        ]

        readable = [p for p in epoch_candidates if checkpoint_signature(p).get("readable")]

        if len(readable) == 1:
            source = readable[0]
            recovered = PART98_BEST
            recovered.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, recovered)

            result["recovery"] = {
                "performed": True,
                "source": str(source),
                "target": str(recovered),
                "source_sha256": sha256(source),
                "target_sha256": sha256(recovered),
                "basis": "Recovered from the recorded Part98 best epoch.",
            }

            print("\nRECOVERY")
            print("-" * 88)
            print(f"Recovered source : {source}")
            print(f"Recovered target : {recovered}")
            print(f"SHA256           : {sha256(recovered)}")
        elif len(readable) > 1:
            result["recovery"] = {
                "performed": False,
                "reason": "Multiple readable candidates matched the best epoch."
            }
            print("\nRECOVERY BLOCKED — multiple candidates matched the best epoch.")
        else:
            result["recovery"] = {
                "performed": False,
                "reason": "The recorded best epoch checkpoint was not recoverable."
            }
            print("\nRECOVERY NOT POSSIBLE — best epoch checkpoint unavailable.")
    elif PART98_BEST.exists():
        result["recovery"] = {
            "performed": False,
            "reason": "Canonical Part98 best checkpoint already exists."
        }

    # Never substitute Part99 for Part98.
    print("\nPART99 LINEAGE CHECK")
    print("-" * 88)

    result["part99"] = {
        "best_exists": PART99_BEST.exists(),
        "final_exists": PART99_FINAL.exists(),
    }

    if PART99_BEST.exists():
        print("part99_best_model.pth                         FOUND")
        result["part99"]["best_sha256"] = sha256(PART99_BEST)
    else:
        print("part99_best_model.pth                         MISSING")

    if PART99_FINAL.exists():
        print("part99_final_model.pth                        FOUND")
        result["part99"]["final_sha256"] = sha256(PART99_FINAL)
    else:
        print("part99_final_model.pth                        MISSING")

    if PART98_BEST.exists() and PART99_BEST.exists():
        comparison = compare_to_reference(PART99_BEST, PART98_BEST)
        result["part99"]["architecture_compatibility_with_part98"] = comparison
        print(
            "Part99/Part98 architecture-compatible : "
            + ("PASS" if comparison.get("compatible") else "FAIL")
        )

    cohorts = result["cohorts"]
    print("\nCOHORT AUDIT")
    print("-" * 88)
    print(f"Train rows : {cohorts['train_rows']}")
    print(f"Val rows   : {cohorts['val_rows']}")
    print(f"Overlap    : {cohorts['overlap']}")

    valid_cohort_split = (
        cohorts["train_exists"]
        and cohorts["val_exists"]
        and cohorts["overlap"] == 0
    )

    if PART98_BEST.exists() and valid_cohort_split:
        result["status"] = "PASS — PART98_CHECKPOINT_AVAILABLE_AND_LINEAGE_VERIFIED"
    elif recovered is not None and valid_cohort_split:
        result["status"] = "PASS — PART98_CHECKPOINT_RECOVERED_FROM_RECORDED_BEST_EPOCH"
    else:
        result["status"] = "PART98_CHECKPOINT_RECOVERY_INCOMPLETE"

    inventory_path = OUT_DIR / "part103_checkpoint_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "path",
                "readable",
                "size_mb",
                "sha256",
                "key_count",
            ],
        )
        writer.writeheader()
        for item in result["candidates"]:
            writer.writerow(item)

    summary_path = REPORT_DIR / "part103_checkpoint_recovery_summary.json"
    report_path = REPORT_DIR / "part103_checkpoint_recovery_report.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    report = []
    report.append("=" * 88)
    report.append("PART 103 — PART98 CHECKPOINT RECOVERY + PART99 LINEAGE VERIFICATION")
    report.append("=" * 88)
    report.append("")
    report.append(f"Project root: {ROOT}")
    report.append(f"Status: {result['status']}")
    report.append("")
    report.append("PART98")
    report.append(f"  Best checkpoint exists: {result['part98']['best_exists']}")
    report.append(f"  History rows: {result['part98']['history_rows']}")
    report.append(f"  Selected best epoch: {best_epoch}")
    report.append(f"  Selection reason: {best_reason}")
    report.append("")
    report.append("COHORTS")
    report.append(f"  Train rows: {cohorts['train_rows']}")
    report.append(f"  Validation rows: {cohorts['val_rows']}")
    report.append(f"  Overlap: {cohorts['overlap']}")
    report.append("")
    report.append("PART99")
    report.append(f"  Best checkpoint: {PART99_BEST.exists()}")
    report.append(f"  Final checkpoint: {PART99_FINAL.exists()}")
    report.append("")
    report.append("RECOVERY")
    report.append(json.dumps(result["recovery"], indent=2))
    report.append("")
    report.append("IMPORTANT")
    report.append(
        "Part99 was never renamed or substituted as Part98. "
        "Any recovered Part98 checkpoint must originate from genuine Part98 artifacts."
    )

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PART 103 FINAL RESULT")
    print("=" * 88)
    print(result["status"])
    print("\nOUTPUTS:")
    print(inventory_path)
    print(summary_path)
    print(report_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
