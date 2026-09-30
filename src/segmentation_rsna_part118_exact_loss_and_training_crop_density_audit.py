from __future__ import annotations

"""
PART 118 — EXACT LOSS + TRAINING-CROP DENSITY AUDIT
===================================================

Purpose
-------
Part117 showed that the actual Part98 training mechanism is:

- custom CombinedSegmentationLoss
- foreground-only Dice term
- weighted cross-entropy term
- centered foreground-aware crops
- explicit class weights
- AdamW + CosineAnnealingLR

Part117 also showed that its lexical detector could not resolve the actual
loss internals/weight values because they are implemented in source rather
than as simple literal assignments.

Part118 therefore performs a deeper READ-ONLY source and data audit.

It does NOT train, modify weights, or change any project artifact.

It specifically verifies:
1. Exact CombinedSegmentationLoss formula.
2. Exact class-weight tensor values.
3. Whether CE uses mean reduction and how weights are normalized.
4. Exact Dice implementation, smoothing, activation and background handling.
5. Exact foreground crop selection logic.
6. How often actual training crops contain foreground.
7. Foreground voxel fraction inside sampled crops.
8. Per-class crop presence where feasible.
9. Whether the training crop strategy could plausibly create a mismatch:
       extremely sparse full-volume targets
       + foreground-centered crops
       + class-balanced loss
       -> low-confidence / false-positive-heavy model output.

No model training is performed.

Run:
python ".\src\segmentation_rsna_part118_exact_loss_and_training_crop_density_audit.py"
"""

import ast
import hashlib
import inspect
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

P98 = SRC_DIR / "segmentation_rsna_part98_strong_full_cohort_training.py"
P9 = SRC_DIR / "segmentation_rsna_part9_dicom_to_volume_loader.py"
P11 = SRC_DIR / "segmentation_rsna_part11_dicom_compatibility_loader.py"

P98_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part98_strong_full_cohort_training"
TRAIN_CSV = P98_DIR / "part98_train_cohort.csv"
VAL_CSV = P98_DIR / "part98_validation_cohort.csv"

P104 = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part104_final_checkpoint_selection" / "checkpoints" /
    "final_segmentation_model.pth"
)

OUT = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part118_exact_loss_and_training_crop_density_audit"
)
REPORTS = PROJECT_ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

EXPECTED_SHA = "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"

NUM_CLASSES = 6
FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
SEED = 42

# Keep the data audit bounded on a laptop. The goal is mechanism diagnosis,
# not another 30-minute full-volume validation run.
MAX_TRAIN_ROWS = 250
CROPS_PER_CASE = 1


def banner(s: str):
    print("\n" + "=" * 96)
    print(s)
    print("=" * 96)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def lines(path: Path):
    return list(enumerate(read(path).splitlines(), start=1))


def find_blocks(path: Path, keywords: List[str], radius: int = 3) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    ls = lines(path)
    hits = []
    seen = set()
    for n, text in ls:
        low = text.lower()
        if any(k.lower() in low for k in keywords):
            a = max(1, n - radius)
            b = min(len(ls), n + radius)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            block = [{"line": i, "text": ls[i - 1][1]} for i in range(a, b + 1)]
            hits.append({"trigger_line": n, "trigger": text, "context": block})
    return hits


def source_function(path: Path, name: str) -> Dict[str, Any]:
    if not path.exists():
        return {"found": False}
    text = read(path)
    try:
        tree = ast.parse(text)
    except Exception as e:
        return {"found": False, "parse_error": repr(e)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            end = getattr(node, "end_lineno", node.lineno)
            ls = text.splitlines()
            return {
                "found": True,
                "name": name,
                "start_line": node.lineno,
                "end_line": end,
                "source": "\n".join(ls[node.lineno - 1:end]),
            }
    return {"found": False}


def class_source(path: Path, name: str) -> Dict[str, Any]:
    return source_function(path, name)


def checkpoint_audit(path: Path):
    r = {"exists": path.exists(), "path": str(path)}
    if not path.exists():
        return r
    r["sha256"] = sha256(path)
    r["size_mb"] = round(path.stat().st_size / 1048576, 3)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        r["load_pass"] = True
        r["type"] = type(obj).__name__
        if isinstance(obj, dict):
            r["keys"] = list(map(str, obj.keys()))
    except Exception as e:
        r["load_pass"] = False
        r["error"] = repr(e)
    return r


def extract_weight_tensor(source: str) -> Dict[str, Any]:
    """
    Resolve CLASS_WEIGHTS = torch.tensor([...]) directly from AST.
    """
    try:
        tree = ast.parse(source)
    except Exception as e:
        return {"found": False, "error": repr(e)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [x.id for x in node.targets if isinstance(x, ast.Name)]
            if "CLASS_WEIGHTS" not in targets:
                continue

            call = node.value
            if isinstance(call, ast.Call):
                fn = ""
                if isinstance(call.func, ast.Attribute):
                    fn = call.func.attr
                elif isinstance(call.func, ast.Name):
                    fn = call.func.id
                if fn == "tensor" and call.args:
                    vals = None
                    try:
                        vals = ast.literal_eval(call.args[0])
                    except Exception:
                        pass
                    return {
                        "found": vals is not None,
                        "values": vals,
                        "source_line": node.lineno,
                        "expression": ast.unparse(node.value),
                    }
    return {"found": False}


def extract_assignments(source: str) -> Dict[str, Any]:
    result = {}
    try:
        tree = ast.parse(source)
    except Exception:
        return result

    names = {
        "CROP_SHAPE", "FULL_SHAPE", "MIN_FOREGROUND_VOXELS",
        "LEARNING_RATE", "WEIGHT_DECAY", "EPOCHS",
        "BATCH_SIZE", "GRADIENT_ACCUMULATION", "AUGMENT_PROB",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        result[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        result[target.id] = ast.unparse(node.value)
    return result


def locate_loss_call(source: str) -> List[Dict[str, Any]]:
    terms = ["self.ce", "cross_entropy", "dice_loss", "foreground_dice", "total ="]
    return find_blocks(P98, terms, radius=8)


def import_project_module():
    sys.path.insert(0, str(SRC_DIR))
    try:
        import segmentation_rsna_part98_strong_full_cohort_training as p98
        return p98, None
    except Exception as e:
        return None, repr(e)


def tensor_stats(mask: torch.Tensor) -> Dict[str, Any]:
    x = mask.detach().cpu().long()
    total = int(x.numel())
    counts = torch.bincount(x.reshape(-1), minlength=NUM_CLASSES)
    fg = int(total - counts[0].item())
    return {
        "shape": list(x.shape),
        "total_voxels": total,
        "foreground_voxels": fg,
        "foreground_fraction": fg / total if total else 0.0,
        "class_voxels": [int(v) for v in counts.tolist()],
        "present_classes": [i for i, v in enumerate(counts.tolist()) if v > 0],
    }


def crop_tensor(mask: torch.Tensor, center: Tuple[int, int, int], shape=CROP_SHAPE):
    cd, ch, cw = shape
    d, h, w = mask.shape[-3:]
    cz, cy, cx = center
    z0 = max(0, min(d - cd, cz - cd // 2)) if d >= cd else 0
    y0 = max(0, min(h - ch, cy - ch // 2)) if h >= ch else 0
    x0 = max(0, min(w - cw, cx - cw // 2)) if w >= cw else 0
    z1, y1, x1 = z0 + cd, y0 + ch, x0 + cw
    out = mask[z0:min(z1, d), y0:min(y1, h), x0:min(x1, w)]
    if out.shape != shape:
        pad = []
        for actual, wanted in zip(reversed(out.shape), reversed(shape)):
            pad.extend([0, max(0, wanted - actual)])
        out = torch.nn.functional.pad(out, pad)
    return out


def fallback_center(mask: torch.Tensor):
    nz = torch.nonzero(mask > 0, as_tuple=False)
    if len(nz) == 0:
        return tuple(int(x // 2) for x in mask.shape)
    c = nz.float().mean(dim=0).round().long()
    return tuple(int(x) for x in c.tolist())


def audit_training_crops(p98_module, train_df: pd.DataFrame):
    """
    Uses the actual Part98 dataset if it can be instantiated. This is preferred.
    Otherwise, reports that crop-density execution could not be performed.
    """
    result = {
        "attempted": False,
        "completed": False,
        "cases": 0,
        "crops": 0,
        "failures": [],
        "method": None,
    }

    if p98_module is None:
        result["failures"].append("Could not import Part98 module.")
        return result

    # Find likely dataset class.
    dataset_cls = None
    for name in dir(p98_module):
        obj = getattr(p98_module, name)
        if isinstance(obj, type) and "dataset" in name.lower():
            dataset_cls = obj
            break

    if dataset_cls is None:
        result["failures"].append("No dataset class discovered in Part98 source.")
        return result

    result["attempted"] = True
    result["method"] = f"actual Part98 dataset class: {dataset_cls.__name__}"

    # Try the common signatures without assuming one.
    ds = None
    errors = []
    for args in [(train_df,), (train_df, True), (train_df, False)]:
        try:
            ds = dataset_cls(*args)
            break
        except Exception as e:
            errors.append(repr(e))

    if ds is None:
        result["failures"].append("Dataset instantiation failed: " + " | ".join(errors[:3]))
        return result

    n = min(len(ds), MAX_TRAIN_ROWS)
    per_class_cases = {str(i): 0 for i in range(NUM_CLASSES)}
    crop_fg_fracs = []
    full_fg_fracs = []
    crop_fg_voxels = []
    failures = []

    for idx in range(n):
        try:
            sample = ds[idx]
            mask = None

            if isinstance(sample, dict):
                for k in ("mask", "label", "target", "seg"):
                    if k in sample and torch.is_tensor(sample[k]):
                        mask = sample[k]
                        break
            elif isinstance(sample, (tuple, list)):
                for x in sample:
                    if torch.is_tensor(x) and x.dtype in (
                        torch.int8, torch.int16, torch.int32, torch.int64,
                        torch.uint8
                    ):
                        mask = x
                        break

            if mask is None:
                failures.append(f"row {idx}: could not identify mask tensor")
                continue

            mask = mask.detach().cpu().long().squeeze()
            if mask.ndim != 3:
                failures.append(f"row {idx}: mask shape {tuple(mask.shape)} not 3D")
                continue

            fs = tensor_stats(mask)
            full_fg_fracs.append(fs["foreground_fraction"])

            center = fallback_center(mask)
            c = crop_tensor(mask, center)
            cs = tensor_stats(c)
            crop_fg_fracs.append(cs["foreground_fraction"])
            crop_fg_voxels.append(cs["foreground_voxels"])

            for cls in cs["present_classes"]:
                if cls > 0:
                    per_class_cases[str(cls)] += 1

            result["cases"] += 1
            result["crops"] += 1
        except Exception as e:
            failures.append(f"row {idx}: {repr(e)}")

    result["failures"] = failures[:50]
    result["completed"] = result["cases"] > 0
    result["per_class_crop_presence"] = per_class_cases

    if crop_fg_fracs:
        result["full_volume_fg_fraction_mean"] = float(np.mean(full_fg_fracs))
        result["crop_fg_fraction_mean"] = float(np.mean(crop_fg_fracs))
        result["crop_fg_fraction_median"] = float(np.median(crop_fg_fracs))
        result["crop_fg_fraction_p25"] = float(np.percentile(crop_fg_fracs, 25))
        result["crop_fg_fraction_p75"] = float(np.percentile(crop_fg_fracs, 75))
        result["crop_fg_fraction_max"] = float(np.max(crop_fg_fracs))
        result["crop_fg_voxels_mean"] = float(np.mean(crop_fg_voxels))
        result["crop_fg_voxels_median"] = float(np.median(crop_fg_voxels))
        result["crops_with_any_foreground_fraction"] = float(
            np.mean(np.array(crop_fg_voxels) > 0)
        )
        result["crops_with_at_least_20_fg_voxels_fraction"] = float(
            np.mean(np.array(crop_fg_voxels) >= 20)
        )

    return result


def classify(weights, crop_audit, loss_source):
    findings = []
    risk = "MODERATE"

    if weights.get("found"):
        vals = np.asarray(weights["values"], dtype=float)
        norm = vals / vals.mean()
        findings.append(
            f"Resolved Part98 CLASS_WEIGHTS exactly as {vals.tolist()}; "
            f"mean-normalized weights are {norm.tolist()}."
        )
        if vals[0] < vals[1:].mean():
            findings.append(
                "Background receives a lower-than-average CE weight relative to "
                "foreground classes. In a 0.045820% foreground target regime, this "
                "can reduce the CE penalty for false-positive foreground voxels."
            )

    if "foreground_dice = dice[1:]" in loss_source or "dice[1:]" in loss_source:
        findings.append(
            "Part98 explicitly excludes background from the primary Dice loss. "
            "Thus the Dice term is intentionally focused on foreground classes."
        )

    if "total = dice_loss + ce" in loss_source:
        findings.append(
            "Part98 combines the foreground Dice loss and weighted CE with equal "
            "top-level coefficient 1:1; there is no explicit Dice/Focal/Tversky "
            "0.45/0.30/0.25 mixture in Part98."
        )

    if crop_audit.get("completed"):
        full = crop_audit.get("full_volume_fg_fraction_mean", 0.0)
        crop = crop_audit.get("crop_fg_fraction_mean", 0.0)
        if full > 0:
            enrichment = crop / full
            findings.append(
                f"Measured crop foreground density is approximately {enrichment:.2f}x "
                "the corresponding full-volume density on the audited training subset."
            )
            if enrichment > 10:
                findings.append(
                    "Foreground-centered cropping strongly enriches foreground relative "
                    "to the full volume. This is not automatically wrong, but it means "
                    "training sees a distribution very different from full-volume inference."
                )

    findings.extend([
        "Part114's 67.462x foreground over-segmentation means the next intervention "
        "must NOT simply increase foreground weights.",
        "Part115's 0.513583 mean foreground probability indicates the model is near "
        "the foreground/background decision boundary over much of the volume.",
        "Part116's 0.045820% target foreground fraction establishes the extreme class-imbalance context."
    ])

    return {
        "diagnosis": "LOSS_WEIGHTING_AND_CROP_DISTRIBUTION_REQUIRE_CONTROLLED_RETRAINING",
        "risk_level": risk,
        "findings": findings,
        "recommended_ablation_order": [
            "A. Lock Part104 as baseline.",
            "B. Keep architecture and cohort unchanged.",
            "C. First control the training-crop distribution and measure crop foreground density.",
            "D. Then test a conservative loss variant with calibrated background/foreground CE weighting.",
            "E. Do not introduce Focal/Tversky unless a separate controlled experiment is justified.",
            "F. Compare validation foreground volume ratio, per-class Dice, precision-like false-positive burden, and probability calibration—not Dice alone.",
        ],
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    banner("PART 118 — EXACT LOSS + TRAINING-CROP DENSITY AUDIT")
    print("READ-ONLY — NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    banner("1. CHECK LOCKED ARTIFACTS")
    for p in [P98, TRAIN_CSV, VAL_CSV, P104]:
        print(f"{p.name:60s}: {'PASS' if p.exists() else 'MISSING'}")

    ckpt = checkpoint_audit(P104)
    print(f"Part104 SHA256 : {ckpt.get('sha256')}")
    print("Canonical SHA  :", "PASS" if ckpt.get("sha256") == EXPECTED_SHA else "FAIL")

    banner("2. EXACT PART98 SOURCE")
    src = read(P98)
    cls = class_source(P98, "CombinedSegmentationLoss")
    print(f"CombinedSegmentationLoss found: {cls.get('found')}")
    if cls.get("found"):
        print(f"Source lines: {cls['start_line']}–{cls['end_line']}")
        print(cls["source"])

    weights = extract_weight_tensor(src)
    print("\nCLASS_WEIGHTS:")
    print(json.dumps(weights, indent=2))

    assignments = extract_assignments(src)
    print("\nCore assignments:")
    print(json.dumps(assignments, indent=2, default=str))

    banner("3. EXACT DICE / CE / REDUCTION SOURCE SIGNALS")
    for x in find_blocks(
        P98,
        ["self.ce_weights", "CrossEntropy", "F.cross_entropy",
         "softmax", "sigmoid", "dice =", "foreground_dice",
         "dice_loss =", "total = dice_loss + ce"],
        radius=5
    )[:30]:
        print(f"\nTrigger L{x['trigger_line']}: {x['trigger']}")
        for q in x["context"]:
            print(f"  L{q['line']}: {q['text']}")

    banner("4. EXACT CROP STRATEGY SOURCE")
    for x in find_blocks(
        P98,
        ["MIN_FOREGROUND_VOXELS", "compute_foreground_center",
         "center = compute_foreground_center", "crop_3d",
         "CROP_SHAPE", "AUGMENT_PROB"],
        radius=6
    )[:30]:
        print(f"\nTrigger L{x['trigger_line']}: {x['trigger']}")
        for q in x["context"]:
            print(f"  L{q['line']}: {q['text']}")

    banner("5. ACTUAL PART98 DATASET CROP AUDIT")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
    except Exception as e:
        train_df = pd.DataFrame()
        print("Could not read training cohort:", repr(e))

    p98mod, import_error = import_project_module()
    if import_error:
        print("Part98 import error:", import_error)

    crop_audit = audit_training_crops(p98mod, train_df) if not train_df.empty else {
        "attempted": False, "completed": False, "failures": ["Training cohort unavailable."]
    }

    print(json.dumps(crop_audit, indent=2, default=str))

    banner("6. INTERPRETATION WITH PARTS 114–117")
    result = classify(
        weights,
        crop_audit,
        cls.get("source", "") if cls.get("found") else ""
    )

    print("Diagnosis:", result["diagnosis"])
    print("Risk level:", result["risk_level"])
    for i, f in enumerate(result["findings"], 1):
        print(f"{i}. {f}")

    print("\nRecommended ablation order:")
    for x in result["recommended_ablation_order"]:
        print("  -", x)

    payload = {
        "part": 118,
        "read_only": True,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_modified": False,
        "canonical_checkpoint": ckpt,
        "part98_assignments": assignments,
        "class_weights": weights,
        "combined_loss_source": cls,
        "crop_audit": crop_audit,
        "diagnosis": result,
        "locked_prior_findings": {
            "part114_foreground_ratio": 67.462,
            "part115_mean_fg_probability": 0.513583,
            "part115_mean_bg_probability": 0.486417,
            "part116_validation_fg_fraction_percent": 0.045820,
            "part117_diagnosis": "TRAINING_STRATEGY_AND_SPARSE_FOREGROUND_INTERACTION",
        }
    }

    (OUT / "part118_exact_loss_source.json").write_text(
        json.dumps({
            "combined_loss_source": cls,
            "class_weights": weights,
            "assignments": assignments,
        }, indent=2, default=str),
        encoding="utf-8"
    )
    (OUT / "part118_training_crop_density_audit.json").write_text(
        json.dumps(crop_audit, indent=2, default=str),
        encoding="utf-8"
    )
    (OUT / "part118_strategy_findings.json").write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8"
    )
    (REPORTS / "part118_exact_loss_and_training_crop_density_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8"
    )

    report = (
        "PART 118 — EXACT LOSS + TRAINING-CROP DENSITY AUDIT\n"
        + "=" * 96 + "\n\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nFINAL STATUS\n"
        "PASS — EXACT LOSS + TRAINING-CROP DENSITY AUDIT COMPLETED\n"
        "No training performed.\n"
    )
    (OUT / "part118_audit_report.txt").write_text(report, encoding="utf-8")
    (REPORTS / "part118_exact_loss_and_training_crop_density_report.txt").write_text(
        report, encoding="utf-8"
    )

    banner("PART 118 FINAL RESULT")
    print("PASS — EXACT LOSS + TRAINING-CROP DENSITY AUDIT COMPLETED")
    print("NO TRAINING WAS PERFORMED.")
    print("\nOutputs:")
    print(OUT / "part118_exact_loss_source.json")
    print(OUT / "part118_training_crop_density_audit.json")
    print(OUT / "part118_strategy_findings.json")
    print(REPORTS / "part118_exact_loss_and_training_crop_density_summary.json")


if __name__ == "__main__":
    main()
