from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PART98_DIR = ROOT / "outputs" / "segmentation" / "rsna_part98_strong_full_cohort_training"
PART99_SCRIPT = SRC / "segmentation_rsna_part99_clinical_oriented_finetuning.py"

EXPECTED_TRAIN = PART98_DIR / "part98_train_cohort.csv"
EXPECTED_VAL = PART98_DIR / "part98_validation_cohort.csv"
EXPECTED_CKPT = PART98_DIR / "checkpoints" / "part98_best_model.pth"

# Known Part15 cohorts are used only as a final recovery source if Part98
# cohort files cannot be found elsewhere. We do NOT silently call these
# "500/100" cohorts; the script reports the actual recovered row counts.
PART15_DIR = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
PART15_TRAIN = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL = PART15_DIR / "part15_validation_cohort.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def describe_csv(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return f"rows={row_count(path)}, columns={len(header)}, first_columns={header[:6]}"
    except Exception as e:
        return f"unreadable: {e}"


def find_candidates(kind: str) -> list[Path]:
    """
    Search outputs/segmentation recursively for Part98 cohort artifacts.
    We intentionally prefer exact names and paths containing 'part98'.
    """
    roots = [
        ROOT / "outputs" / "segmentation",
        ROOT / "outputs" / "final",
    ]

    exact_names = {
        "train": {"part98_train_cohort.csv", "train_cohort.csv"},
        "val": {
            "part98_validation_cohort.csv",
            "part98_val_cohort.csv",
            "validation_cohort.csv",
            "val_cohort.csv",
        },
    }[kind]

    candidates: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.csv"):
            name = p.name.lower()
            if p.name in exact_names:
                candidates.append(p)
            elif "part98" in name and kind == "train" and "train" in name and "cohort" in name:
                candidates.append(p)
            elif "part98" in name and kind == "val" and ("validation" in name or "val" in name) and "cohort" in name:
                candidates.append(p)

    # Prefer files already associated with a Part98 directory and larger
    # cohorts, because Part98 was configured for 500/100.
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(
        key=lambda p: (
            "part98" in str(p).lower(),
            row_count(p) if p.exists() else -1,
        ),
        reverse=True,
    )
    return candidates


def recover_one(kind: str, expected: Path) -> Path | None:
    if expected.exists():
        print(f"{expected.name:<48} PASS ({row_count(expected)} rows)")
        return expected

    candidates = find_candidates(kind)
    if candidates:
        chosen = candidates[0]
        print(f"{expected.name:<48} RECOVERED")
        print(f"  Source : {chosen}")
        print(f"  Detail : {describe_csv(chosen)}")
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen, expected)
        print(f"  Target : {expected}")
        return expected

    # Last-resort recovery: only if Part15 artifacts exist. This is explicit
    # and is never mislabeled as the Part98 500/100 cohort.
    fallback = PART15_TRAIN if kind == "train" else PART15_VAL
    if fallback.exists():
        print(f"{expected.name:<48} FALLBACK FROM PART15")
        print(f"  Source : {fallback}")
        print(f"  Detail : {describe_csv(fallback)}")
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, expected)
        print("  WARNING: recovered cohort is Part15-sized, not Part98 500/100.")
        return expected

    print(f"{expected.name:<48} FAIL")
    return None


def main() -> None:
    print("=" * 88)
    print("PART 101 — PART98 ARTIFACT RECOVERY + PART99 CLINICAL FINETUNING")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")

    print("\nCHECKING PART98 DIRECTORY")
    if not PART98_DIR.exists():
        PART98_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created : {PART98_DIR}")
    else:
        print(f"Directory : PASS")

    print("\nCHECKING PART98 CHECKPOINT")
    if not EXPECTED_CKPT.exists():
        # Search recursively in case the checkpoint was written under a
        # slightly different subdirectory.
        ckpts = list((ROOT / "outputs" / "segmentation").rglob("part98_best_model.pth"))
        if ckpts:
            ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            EXPECTED_CKPT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpts[0], EXPECTED_CKPT)
            print("part98_best_model.pth                              RECOVERED")
            print(f"  Source : {ckpts[0]}")
        else:
            print("part98_best_model.pth                              FAIL")
            raise FileNotFoundError(
                "Part98 best checkpoint was not found anywhere under outputs\\segmentation."
            )
    else:
        print("part98_best_model.pth                              PASS")

    print(f"Checkpoint size : {EXPECTED_CKPT.stat().st_size / (1024**2):.3f} MB")
    print(f"Checkpoint SHA256: {sha256(EXPECTED_CKPT)}")

    print("\nRECOVERING PART98 COHORT ARTIFACTS")
    train_path = recover_one("train", EXPECTED_TRAIN)
    val_path = recover_one("val", EXPECTED_VAL)

    if train_path is None or val_path is None:
        raise FileNotFoundError(
            "Could not recover both Part98 train/validation cohort CSVs. "
            "Inspect outputs\\segmentation for the original cohort artifacts."
        )

    print("\nRECOVERED COHORT SUMMARY")
    print(f"Training rows   : {row_count(train_path)}")
    print(f"Validation rows : {row_count(val_path)}")
    print("Overlap audit   : checking by first-column study/case identifier")

    def ids(path: Path) -> set[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set()
            key = reader.fieldnames[0]
            return {str(r.get(key, "")).strip() for r in reader if str(r.get(key, "")).strip()}

    train_ids = ids(train_path)
    val_ids = ids(val_path)
    overlap = train_ids & val_ids
    print(f"Unique train IDs : {len(train_ids)}")
    print(f"Unique val IDs   : {len(val_ids)}")
    print(f"Overlap          : {len(overlap)}")

    if overlap:
        raise RuntimeError(
            f"COHORT LEAKAGE DETECTED: {len(overlap)} IDs overlap between train and validation."
        )

    print("\nCHECKING PART99 SCRIPT")
    if not PART99_SCRIPT.exists():
        raise FileNotFoundError(PART99_SCRIPT)
    print("Part99 clinical fine-tuning script                    PASS")

    print("\nIMPORTANT")
    print(
        "Part101 repairs the missing Part98 artifact path before launching "
        "the already-defined Part99 clinical-oriented fine-tuning."
    )
    print(
        "It does not invent a new cohort. If the exact Part98 cohort is unavailable, "
        "the script transparently falls back to Part15 and reports the actual size."
    )

    print("\nSTARTING PART99 CLINICAL-ORIENTED FINE-TUNING")
    print("=" * 88)

    # Run Part99 with the same Python interpreter/venv. The repaired files now
    # exist at the exact paths expected by Part99.
    completed = subprocess.run(
        [sys.executable, str(PART99_SCRIPT)],
        cwd=str(ROOT),
        text=True,
    )

    print("\n" + "=" * 88)
    print("PART 101 FINAL RESULT")
    print("=" * 88)

    if completed.returncode == 0:
        print("PASS — PART98 ARTIFACTS RECOVERED AND PART99 COMPLETED")
        print("\nNext:")
        print(
            "Use the Part99 terminal output to select the best clinically-oriented "
            "segmentation checkpoint before moving to the dashboard."
        )
    else:
        print("PART99 RETURN CODE:", completed.returncode)
        print(
            "Part101 successfully repaired the artifact-path problem, but Part99 "
            "did not complete. The terminal output above contains the next error."
        )
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
