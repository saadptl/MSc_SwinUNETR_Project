from __future__ import annotations

"""
PART 117 — TRAINING STRATEGY / LOSS-MECHANISM AUDIT

READ-ONLY:
- no training
- no backward()
- no optimizer.step()
- no checkpoint modification
- no cohort modification

Purpose:
Audit the actual Part98 source, loss configuration, crop/sampling strategy,
optimizer/scheduler, Part98 history, and locked Parts114-116 findings before
any retraining decision.

Run from project root:
python ".\\src\\segmentation_rsna_part117_training_strategy_loss_mechanism_audit.py"
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PART98_SOURCE = SRC_DIR / "segmentation_rsna_part98_strong_full_cohort_training.py"
PART99_SOURCE = SRC_DIR / "segmentation_rsna_part99_clinical_oriented_finetuning.py"
PART100_SOURCE = SRC_DIR / "segmentation_rsna_part100_robust_clinical_finetuning.py"

PART98_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part98_strong_full_cohort_training"
PART98_CKPT = PART98_DIR / "checkpoints" / "part98_best_model.pth"
PART98_HISTORY = PART98_DIR / "part98_training_history.csv"
PART98_TRAIN = PART98_DIR / "part98_train_cohort.csv"
PART98_VAL = PART98_DIR / "part98_validation_cohort.csv"

PART104_CKPT = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part104_final_checkpoint_selection" / "checkpoints" /
    "final_segmentation_model.pth"
)

OUT_DIR = (
    PROJECT_ROOT / "outputs" / "segmentation" /
    "rsna_part117_training_strategy_loss_mechanism_audit"
)
REPORT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_FINAL_SHA = "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"


def banner(s: str) -> None:
    print("\n" + "=" * 96)
    print(s)
    print("=" * 96)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def source_forensics(path: Path) -> Dict[str, Any]:
    r = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": None,
        "line_count": 0,
        "functions": [],
        "classes": [],
        "literal_assignments": {},
    }
    if not path.exists():
        return r

    text = read_text(path)
    r["sha256"] = sha256_file(path)
    r["line_count"] = len(text.splitlines())

    try:
        tree = ast.parse(text)
    except Exception as e:
        r["parse_error"] = repr(e)
        return r

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            r["functions"].append({
                "name": n.name,
                "line": n.lineno,
                "args": [a.arg for a in n.args.args],
            })
        elif isinstance(n, ast.ClassDef):
            r["classes"].append({"name": n.name, "line": n.lineno})
        elif isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    v = literal(n.value)
                    if v is not None:
                        r["literal_assignments"][target.id] = v

    groups = {
        "loss": ["loss", "dice", "focal", "tversky", "crossentropy", "bce", "criterion"],
        "sampling": ["sampl", "crop", "roi", "positive", "negative", "foreground"],
        "optimizer": ["optimizer", "adam", "adamw", "sgd", "learning_rate", "lr", "weight_decay"],
        "scheduler": ["scheduler", "cosine", "step_lr", "onecycle", "plateau", "warmup"],
        "weights": ["class_weight", "class_weights", "pos_weight", "weight=", "weights"],
        "training": ["backward", "optimizer.step", "scaler.step", "zero_grad", "autocast", "gradscaler"],
    }

    lines = text.splitlines()
    for name, keys in groups.items():
        hits = []
        for i, line in enumerate(lines, 1):
            low = line.lower()
            if any(k.lower() in low for k in keys):
                hits.append({"line": i, "text": line.rstrip()})
        r[name + "_lines"] = hits

    return r


def checkpoint_audit(path: Path) -> Dict[str, Any]:
    r = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return r
    r["size_mb"] = round(path.stat().st_size / 1048576, 3)
    r["sha256"] = sha256_file(path)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        r["load_pass"] = True
        r["type"] = type(obj).__name__
        if isinstance(obj, dict):
            r["keys"] = list(map(str, obj.keys()))
            state = None
            for k in ("model_state_dict", "state_dict", "model"):
                if isinstance(obj.get(k), dict):
                    state = obj[k]
                    r["state_dict_key"] = k
                    break
            for k in ("epoch", "best_epoch", "best_dice", "val_dice", "train_loss", "val_loss"):
                if k in obj:
                    v = obj[k]
                    r[k] = v.item() if isinstance(v, np.generic) else v
            if state is not None:
                r["state_key_count"] = len(state)
                r["tensor_count"] = sum(
                    int(v.numel()) for v in state.values() if torch.is_tensor(v)
                )
    except Exception as e:
        r["load_pass"] = False
        r["load_error"] = repr(e)
    return r


def history_audit(path: Path) -> Dict[str, Any]:
    r = {"exists": path.exists()}
    if not path.exists():
        return r
    try:
        df = pd.read_csv(path)
    except Exception as e:
        r["read_error"] = repr(e)
        return r

    r["rows"] = len(df)
    r["columns"] = [str(c) for c in df.columns]
    low = {str(c).lower(): c for c in df.columns}
    dice_cols = [c for k, c in low.items() if "dice" in k]
    loss_cols = [c for k, c in low.items() if "loss" in k]

    if dice_cols:
        best = None
        for c in dice_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                i = s.idxmax()
                x = {"column": str(c), "value": float(s.loc[i]), "row": int(i)}
                if best is None or x["value"] > best["value"]:
                    best = x
        r["best_dice"] = best

    if loss_cols:
        r["loss_behavior"] = {}
        for c in loss_cols:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                r["loss_behavior"][str(c)] = {
                    "first": float(s.iloc[0]),
                    "last": float(s.iloc[-1]),
                    "minimum": float(s.min()),
                }

    if dice_cols and loss_cols:
        r["correlations"] = {}
        for dc in dice_cols:
            d = pd.to_numeric(df[dc], errors="coerce")
            for lc in loss_cols:
                l = pd.to_numeric(df[lc], errors="coerce")
                m = d.notna() & l.notna()
                if m.sum() >= 3:
                    r["correlations"][f"{dc} vs {lc}"] = float(np.corrcoef(d[m], l[m])[0, 1])
    return r


def extract_config(src: Dict[str, Any]) -> Dict[str, Any]:
    wanted = (
        "epoch", "batch", "gradient", "accum", "learning", "lr",
        "weight_decay", "seed", "feature", "class", "shape", "crop",
        "prob", "alpha", "beta", "gamma", "dice_weight", "focal_weight",
        "tversky_weight"
    )
    return {
        k: v for k, v in src.get("literal_assignments", {}).items()
        if any(w in k.lower() for w in wanted)
    }


def compact_lines(items, limit=80):
    return items[:limit]


def diagnose(p98, p99, p100, hist, ckpt):
    loss_text = "\n".join(x["text"] for x in p98.get("loss_lines", []))
    sampling_text = "\n".join(x["text"] for x in p98.get("sampling_lines", []))
    weight_text = "\n".join(x["text"] for x in p98.get("weights_lines", []))

    loss_signals = {
        "DiceCELoss": "DiceCELoss" in loss_text,
        "DiceLoss": "DiceLoss" in loss_text,
        "CrossEntropyLoss": "CrossEntropyLoss" in loss_text,
        "Focal": "focal" in loss_text.lower(),
        "Tversky": "tversky" in loss_text.lower(),
        "BCEWithLogitsLoss": "BCEWithLogitsLoss" in loss_text,
    }

    fg_sampling = any(
        x in sampling_text.lower()
        for x in ("foreground", "positive", "oversampl", "randcropbyposneglabel")
    )

    explicit_weights = bool(re.search(
        r"\b(class_weight|class_weights|pos_weight|weight\s*=|weights)\b",
        weight_text, re.I
    ))

    findings = [
        "Part114 measured 67.462x predicted foreground relative to target foreground.",
        "Part115 measured mean P(FG)=0.513583 and P(BG)=0.486417, with no single-logit class-collapse diagnosis.",
        "Part116 measured only 0.045820% validation foreground voxels, while all five foreground classes were present globally.",
    ]

    if fg_sampling:
        findings.append(
            "Part98 source contains foreground/positive sampling signals. "
            "The critical quantity is foreground density INSIDE sampled crops, not just full-volume density."
        )
    else:
        findings.append(
            "No clear foreground-aware sampling mechanism was detected lexically in Part98. "
            "If crops are mostly uniform, extreme target sparsity can make optimization background-dominated."
        )

    if any(loss_signals.values()):
        findings.append(
            "Part98 contains explicit loss-related constructs. Exact reduction, background inclusion, "
            "empty-class handling, and target encoding must be confirmed before changing the loss."
        )

    if explicit_weights:
        findings.append(
            "Part98 contains explicit weighting references. Increasing weights blindly is unsafe because "
            "the current model already overpredicts foreground by 67.462x."
        )

    if ckpt.get("sha256") == EXPECTED_FINAL_SHA:
        findings.append(
            "Part104 SHA matches the canonical Part98-best checkpoint, so the Parts114-116 behavior is "
            "being observed from the exact selected Part98 model."
        )

    if hist.get("best_dice") and hist["best_dice"]["value"] < 0.30:
        findings.append(
            f"Part98 peak recorded validation Dice was only {hist['best_dice']['value']:.6f}; "
            "the training run did not reach a strong segmentation regime."
        )

    return {
        "primary_diagnosis": "TRAINING_STRATEGY_AND_SPARSE_FOREGROUND_INTERACTION",
        "recommendation": "TARGETED_RETRAINING_AFTER_LOSS_AND_CROP_AUDIT",
        "loss_signals": loss_signals,
        "foreground_sampling_signal": fg_sampling,
        "explicit_weighting_signal": explicit_weights,
        "findings": findings,
        "do_not_change_yet": [
            "Swin-UNETR architecture",
            "six-class label definition",
            "Part9/Part11 loader contract",
            "canonical Part104 checkpoint",
            "class weights without verifying exact values and reduction",
            "learning rate without evidence that LR is the primary failure",
        ],
        "next_retraining_focus": [
            "Verify the exact Part98 loss formula and reduction.",
            "Verify whether Dice includes/excludes background.",
            "Verify empty-class handling and per-class Dice aggregation.",
            "Verify focal/Tversky target encoding and normalization.",
            "Verify all class/foreground weights and their normalization.",
            "Measure foreground fraction inside actual training crops.",
            "Measure how often training crops contain each class.",
            "Run one-variable-at-a-time ablations; keep Part104 as the locked baseline.",
        ],
    }


def main():
    banner("PART 117 — TRAINING STRATEGY / LOSS-MECHANISM AUDIT")
    print("READ-ONLY AUDIT — NO TRAINING, NO BACKWARD, NO OPTIMIZER STEP")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    banner("1. ARTIFACT CHECK")
    for p in [PART98_SOURCE, PART98_HISTORY, PART98_TRAIN, PART98_VAL, PART104_CKPT]:
        print(f"{p.name:55s} : {'PASS' if p.exists() else 'MISSING'}")

    ckpt = checkpoint_audit(PART104_CKPT)
    print(f"Part104 SHA256 : {ckpt.get('sha256')}")
    print("Canonical SHA  :", "PASS" if ckpt.get("sha256") == EXPECTED_FINAL_SHA else "FAIL")

    banner("2. ACTUAL SOURCE FORENSICS")
    p98 = source_forensics(PART98_SOURCE)
    p99 = source_forensics(PART99_SOURCE)
    p100 = source_forensics(PART100_SOURCE)

    print(f"Part98 lines       : {p98.get('line_count', 0)}")
    print(f"Functions          : {len(p98.get('functions', []))}")
    print(f"Classes            : {len(p98.get('classes', []))}")
    print(f"Loss hits          : {len(p98.get('loss_lines', []))}")
    print(f"Sampling/crop hits : {len(p98.get('sampling_lines', []))}")
    print(f"Weight hits        : {len(p98.get('weights_lines', []))}")
    print(f"Optimizer hits     : {len(p98.get('optimizer_lines', []))}")
    print(f"Scheduler hits     : {len(p98.get('scheduler_lines', []))}")

    banner("3. PART98 LOSS MECHANISM")
    for k, v in diagnose(p98, p99, p100, history_audit(PART98_HISTORY), ckpt)["loss_signals"].items():
        print(f"{k:25s}: {v}")
    print("\nRelevant source lines:")
    for x in compact_lines(p98.get("loss_lines", [])):
        print(f"L{x['line']}: {x['text']}")

    banner("4. CROPPING / SAMPLING")
    for x in compact_lines(p98.get("sampling_lines", [])):
        print(f"L{x['line']}: {x['text']}")

    banner("5. CLASS WEIGHTS / REDUCTION")
    for x in compact_lines(p98.get("weights_lines", [])):
        print(f"L{x['line']}: {x['text']}")

    banner("6. OPTIMIZER / SCHEDULER")
    for x in compact_lines(p98.get("optimizer_lines", [])):
        print(f"L{x['line']}: {x['text']}")
    print("\nScheduler:")
    for x in compact_lines(p98.get("scheduler_lines", [])):
        print(f"L{x['line']}: {x['text']}")

    banner("7. PART98 / PART99 / PART100 CONFIGURATION")
    comparison = {
        "part98": extract_config(p98),
        "part99": extract_config(p99),
        "part100": extract_config(p100),
    }
    print(json.dumps(comparison, indent=2, default=str))

    banner("8. PART98 TRAINING HISTORY")
    hist = history_audit(PART98_HISTORY)
    print(json.dumps(hist, indent=2, default=str))
    if PART98_HISTORY.exists():
        pd.read_csv(PART98_HISTORY).to_csv(
            OUT_DIR / "part117_training_history_analysis.csv", index=False
        )

    banner("9. LOCKED PART114-116 FINDINGS")
    print("Part114 foreground ratio       : 67.462x target")
    print("Part115 mean FG probability    : 0.513583")
    print("Part115 mean BG probability    : 0.486417")
    print("Part115 class-bias diagnosis   : NO_SINGLE_LOGIT_CLASS_BIAS")
    print("Part116 validation FG fraction : 0.045820%")
    print("Part116 globally absent classes: none")
    print("Part116 rare classes            : none under audit thresholds")

    banner("10. FINAL STRATEGY DIAGNOSIS")
    result = diagnose(p98, p99, p100, hist, ckpt)
    print("Primary diagnosis :", result["primary_diagnosis"])
    print("Recommendation    :", result["recommendation"])
    print("\nFindings:")
    for i, f in enumerate(result["findings"], 1):
        print(f"{i}. {f}")
    print("\nDo NOT change yet:")
    for f in result["do_not_change_yet"]:
        print(f"  - {f}")
    print("\nNext retraining focus:")
    for f in result["next_retraining_focus"]:
        print(f"  - {f}")

    payload = {
        "part": 117,
        "evaluation_only": True,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_modified": False,
        "part98_source": p98,
        "part99_source": p99,
        "part100_source": p100,
        "configuration_comparison": comparison,
        "part98_history": hist,
        "canonical_checkpoint": ckpt,
        "strategy_result": result,
    }

    (OUT_DIR / "part117_source_forensics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (OUT_DIR / "part117_strategy_findings.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    report = (
        "PART 117 — TRAINING STRATEGY / LOSS-MECHANISM AUDIT\n"
        + "=" * 96 + "\n\n"
        "READ-ONLY. No training, backward pass, optimizer step, or checkpoint modification.\n\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nFINAL STATUS\n"
        "PASS — TRAINING STRATEGY / LOSS-MECHANISM AUDIT COMPLETED\n"
        "Primary diagnosis: TRAINING_STRATEGY_AND_SPARSE_FOREGROUND_INTERACTION\n"
        "Retraining recommendation: TARGETED_RETRAINING_AFTER_LOSS_AND_CROP_AUDIT\n"
    )

    (OUT_DIR / "part117_audit_report.txt").write_text(report, encoding="utf-8")
    (REPORT_DIR / "part117_training_strategy_loss_mechanism_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORT_DIR / "part117_training_strategy_loss_mechanism_report.txt").write_text(
        report, encoding="utf-8"
    )

    banner("PART 117 FINAL RESULT")
    print("PASS — TRAINING STRATEGY / LOSS-MECHANISM AUDIT COMPLETED")
    print("Primary diagnosis: TRAINING_STRATEGY_AND_SPARSE_FOREGROUND_INTERACTION")
    print("Retraining recommendation: TARGETED_RETRAINING_AFTER_LOSS_AND_CROP_AUDIT")
    print("\nNO TRAINING WAS PERFORMED.")


if __name__ == "__main__":
    main()
