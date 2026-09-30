"""
PHASE 4 - PART 28
RSNA-ONLY DICECELoss TARGET-ENCODING / FOREGROUND-GRADIENT AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Part 27 established that:
  * training labels contain foreground voxels;
  * validation labels contain foreground voxels;
  * the Part 15 checkpoint strongly predicts background;
  * the Part 27 DiceCELoss gradient experiment was invalid because the
    integer target shape was not compatible with the installed loss setup.

Part 28 fixes that experiment and answers:
  1. What target encoding does the exact Part 11 DiceCELoss configuration require?
  2. Does the loss produce finite foreground gradients for sparse labels?
  3. How large are CE and Dice gradients for foreground/background logits?
  4. Does extreme foreground sparsity make the foreground signal negligible?
  5. Is the loss implementation itself a plausible cause of collapse?

The audit uses synthetic tensors for controlled gradient experiments and
optionally inspects real Part 15 validation labels. It NEVER calls backward()
on the real model and NEVER changes checkpoint parameters.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)
PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
PART15_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "best_model.pth"
)
PART15_VAL = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)
PART8_VAL = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part28_diceceloss_foreground_gradient_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_float(x: Any) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float("nan")


def validate_paths() -> None:
    banner("PATH VALIDATION")
    paths = {
        "RSNA root": DATASET_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 validation cohort": PART15_VAL,
        "Part 8 validation manifest": PART8_VAL,
    }

    missing = []
    for name, path in paths.items():
        ok = path.exists()
        print(f"{name:<36}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 28 input(s):\n" + "\n".join(missing)
        )


def import_part11():
    banner("IMPORTING VALIDATED PART 11")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_part28",
        PART11_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {PART11_SOURCE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    print("✓ Corrected Part 11 imported.")

    required = ["create_model", "preprocess_case", "dice_from_prediction"]
    for name in required:
        if not hasattr(module, name):
            raise AttributeError(f"Part 11 missing required API: {name}")
        print(f"{name:<22}: {inspect.signature(getattr(module, name))}")

    return module


def import_diceceloss():
    from monai.losses import DiceCELoss

    return DiceCELoss


def inspect_loss_configuration(part11) -> Dict[str, Any]:
    banner("INSPECTING EXACT DICECELOSS CONFIGURATION")

    source = inspect.getsource(part11)
    lines = source.splitlines()

    matches = []
    for i, line in enumerate(lines):
        if "DiceCELoss" in line:
            start = max(0, i - 8)
            end = min(len(lines), i + 18)
            matches.append("\n".join(
                f"{j + 1:04d}: {lines[j]}"
                for j in range(start, end)
            ))

    print("Part 11 DiceCELoss-related source:")
    if matches:
        print("\n---\n".join(matches[:5]))
    else:
        print("No DiceCELoss constructor text found by source inspection.")

    DiceCELoss = import_diceceloss()

    # Reconstruct the configuration used by Part 11 where possible.
    # Part 11's printed configuration in previous audits identifies
    # DiceCELoss as the training/evaluation loss. The exact MONAI defaults
    # are intentionally retained here unless source inspection discovers
    # explicit constructor arguments.
    try:
        loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )
        config = {
            "include_background": True,
            "to_onehot_y": True,
            "softmax": True,
            "source": "Part 28 controlled reconstruction",
        }
    except TypeError:
        loss = DiceCELoss()
        config = {
            "source": "MONAI default constructor fallback",
        }

    print("\nControlled loss configuration:")
    for k, v in config.items():
        print(f"{k:<36}: {v}")

    return {
        "loss": loss,
        "config": config,
        "source_matches": matches[:5],
        "constructor_signature": str(inspect.signature(DiceCELoss)),
    }


def tensor_shapes() -> Dict[str, Any]:
    banner("TARGET ENCODING SANITY CHECK")

    b, d, h, w = 1, 8, 12, 12

    integer_target = torch.zeros(
    (b, 1, d, h, w),
    dtype=torch.long,
)

    integer_target[:, 0, 2:4, 4:8, 4:8] = 1

    one_hot = F.one_hot(
    integer_target[:, 0],
    num_classes=NUM_CLASSES,
).permute(0, 4, 1, 2, 3).float()

    logits = torch.zeros(
        (b, NUM_CLASSES, d, h, w),
        dtype=torch.float32,
    )

    print(f"Logits shape                         : {tuple(logits.shape)}")
    print(f"Integer target shape                 : {tuple(integer_target.shape)}")
    print(f"One-hot target shape                 : {tuple(one_hot.shape)}")
    print(f"Integer target dtype                 : {integer_target.dtype}")
    print(f"One-hot target dtype                 : {one_hot.dtype}")

    return {
        "logits_shape": list(logits.shape),
        "integer_target_shape": list(integer_target.shape),
        "one_hot_target_shape": list(one_hot.shape),
        "integer_target_dtype": str(integer_target.dtype),
        "one_hot_target_dtype": str(one_hot.dtype),
    }


def make_sparse_target(
    shape: Tuple[int, int, int, int],
    foreground_voxels: int,
    class_id: int = 1,
) -> torch.Tensor:
    """
    Create a MONAI-compatible integer segmentation target.

    Input shape:
        (B, D, H, W)

    Returned shape:
        (B, 1, D, H, W)

    This singleton channel is required by the installed MONAI 1.6.0
    DiceCELoss when to_onehot_y=True.
    """
    b, d, h, w = shape

    target = torch.zeros(
        (b, 1, d, h, w),
        dtype=torch.long,
    )

    total = b * d * h * w
    n = max(0, min(int(foreground_voxels), total))

    if n > 0:
        flat = target[:, 0].reshape(-1)
        flat[:n] = int(class_id)

    return target


def gradient_summary(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
) -> Dict[str, Any]:
    x = logits.clone().detach().requires_grad_(True)

    loss = loss_fn(x, target)
    if loss.ndim != 0:
        loss = loss.mean()

    loss.backward()

    g = x.grad.detach()

    result = {
        "loss": safe_float(loss.item()),
        "grad_abs_mean": safe_float(g.abs().mean().item()),
        "grad_abs_max": safe_float(g.abs().max().item()),
        "grad_l2": safe_float(torch.linalg.vector_norm(g).item()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_grad": bool(torch.isfinite(g).all()),
    }

    per_class = []
    for c in range(g.shape[1]):
        gc = g[:, c]
        per_class.append({
            "class_id": c,
            "class_name": CLASS_NAMES.get(c, f"Class {c}"),
            "grad_abs_mean": safe_float(gc.abs().mean().item()),
            "grad_abs_max": safe_float(gc.abs().max().item()),
            "grad_l2": safe_float(torch.linalg.vector_norm(gc).item()),
        })

    result["per_class"] = per_class
    return result


def run_gradient_scenarios(loss_fn) -> List[Dict[str, Any]]:
    banner("RUNNING CONTROLLED DICECELOSS GRADIENT SCENARIOS")

    shape = (1, 8, 12, 12)
    scenarios = [
        ("all_background", 0, 1),
        ("one_foreground_voxel", 1, 1),
        ("sparse_10_voxels", 10, 1),
        ("sparse_100_voxels", 100, 1),
        ("moderate_500_voxels", 500, 1),
        ("class_2_sparse", 10, 2),
        ("class_3_sparse", 10, 3),
        ("class_4_sparse", 10, 4),
        ("class_5_sparse", 10, 5),
    ]

    rows = []
    for name, fg, cls in scenarios:
        target = make_sparse_target(shape, fg, cls)
        logits = torch.zeros(
            (shape[0], NUM_CLASSES, shape[1], shape[2], shape[3]),
            dtype=torch.float32,
        )

        try:
            result = gradient_summary(logits, target, loss_fn)
            row = {
                "scenario": name,
                "foreground_voxels": fg,
                "foreground_class": cls,
                "loss": result["loss"],
                "grad_abs_mean": result["grad_abs_mean"],
                "grad_abs_max": result["grad_abs_max"],
                "grad_l2": result["grad_l2"],
                "finite_loss": result["finite_loss"],
                "finite_grad": result["finite_grad"],
            }

            for item in result["per_class"]:
                c = item["class_id"]
                row[f"class_{c}_grad_abs_mean"] = item["grad_abs_mean"]

            rows.append(row)

            print(
                f"{name:<26} fg={fg:<5} class={cls} "
                f"loss={result['loss']:.8f} "
                f"grad_mean={result['grad_abs_mean']:.8e} "
                f"grad_max={result['grad_abs_max']:.8e}"
            )
        except Exception as exc:
            print(f"{name:<26} ERROR: {repr(exc)}")
            rows.append({
                "scenario": name,
                "foreground_voxels": fg,
                "foreground_class": cls,
                "error": repr(exc),
            })

    return rows


def run_probability_gradient_scenarios(loss_fn) -> List[Dict[str, Any]]:
    banner("FOREGROUND VS BACKGROUND LOGIT GRADIENT AUDIT")

    shape = (1, 8, 12, 12)
    target = make_sparse_target(shape, 10, 1)

    # Controlled logits: background initially dominates, mimicking Part 15.
    cases = [
        ("uniform_logits", 0.0),
        ("background_plus_1", 1.0),
        ("background_plus_2", 2.0),
        ("background_plus_4", 4.0),
    ]

    rows = []

    for name, bg_bias in cases:
        logits = torch.zeros(
            (1, NUM_CLASSES, shape[1], shape[2], shape[3]),
            dtype=torch.float32,
        )
        logits[:, 0] = bg_bias

        result = gradient_summary(logits, target, loss_fn)
        pc = {x["class_id"]: x for x in result["per_class"]}

        fg_grad = pc[1]["grad_abs_mean"]
        bg_grad = pc[0]["grad_abs_mean"]

        ratio = (
            fg_grad / bg_grad
            if bg_grad > 0
            else float("nan")
        )

        row = {
            "scenario": name,
            "background_logit_bias": bg_bias,
            "loss": result["loss"],
            "background_probability": float(
                torch.softmax(logits, dim=1)[:, 0].mean().item()
            ),
            "foreground_class_1_probability": float(
                torch.softmax(logits, dim=1)[:, 1].mean().item()
            ),
            "background_grad_abs_mean": bg_grad,
            "foreground_class_1_grad_abs_mean": fg_grad,
            "foreground_to_background_gradient_ratio": ratio,
        }
        rows.append(row)

        print(
            f"{name:<22} bg_prob={row['background_probability']:.6f} "
            f"fg_prob={row['foreground_class_1_probability']:.6f} "
            f"bg_grad={bg_grad:.8e} "
            f"fg_grad={fg_grad:.8e} "
            f"fg/bg={ratio:.6f}"
        )

    return rows


def audit_real_target_encoding() -> Dict[str, Any]:
    banner("REAL PART 15 LABEL ENCODING CHECK")

    if not PART15_VAL.exists():
        return {"status": "SKIPPED", "reason": "Part 15 validation cohort missing"}

    df = pd.read_csv(PART15_VAL)

    result = {
        "status": "OK",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "foreground_voxel_fields": {},
    }

    candidate_columns = [
        c for c in df.columns
        if any(
            token in str(c).lower()
            for token in ["mask", "foreground", "voxel", "positive"]
        )
    ]

    print(f"Part 15 validation rows               : {len(df)}")
    print(f"Candidate label-related columns      : {candidate_columns}")

    for col in candidate_columns:
        try:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                result["foreground_voxel_fields"][col] = {
                    "non_null": int(values.notna().sum()),
                    "min": safe_float(values.min()),
                    "max": safe_float(values.max()),
                    "mean": safe_float(values.mean()),
                }
        except Exception:
            continue

    return result


def test_part11_dice_api_consistency(part11) -> Dict[str, Any]:
    banner("PART 11 DICE API CROSS-CHECK")

    logits = torch.zeros(
        (1, NUM_CLASSES, 8, 12, 12),
        dtype=torch.float32,
    )
    target = make_sparse_target((1, 8, 12, 12), 10, 1)

    try:
        value = part11.dice_from_prediction(logits, target)
        print(f"Part 11 Dice result type             : {type(value)}")
        print(f"Part 11 Dice result                  : {value}")
        return {
            "status": "OK",
            "result": value,
        }
    except Exception as exc:
        print(f"Part 11 Dice API test ERROR: {repr(exc)}")
        return {
            "status": "ERROR",
            "error": repr(exc),
        }


def save_results(
    gradient_rows: List[Dict[str, Any]],
    probability_rows: List[Dict[str, Any]],
    shape_info: Dict[str, Any],
    real_label_info: Dict[str, Any],
    dice_api_info: Dict[str, Any],
    loss_info: Dict[str, Any],
) -> None:
    banner("SAVING PART 28 RESULTS")

    pd.DataFrame(gradient_rows).to_csv(
        OUTPUT_DIR / "part28_diceceloss_gradient_scenarios.csv",
        index=False,
    )

    pd.DataFrame(probability_rows).to_csv(
        OUTPUT_DIR / "part28_foreground_background_gradient_comparison.csv",
        index=False,
    )

    summary = {
        "phase": "PHASE 4 - PART 28",
        "purpose": "DiceCELoss target encoding and foreground gradient audit",
        "evaluation_only": True,
        "training_performed": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "shape_info": shape_info,
        "real_label_info": real_label_info,
        "dice_api_info": dice_api_info,
        "loss_config": loss_info["config"],
        "loss_signature": loss_info["constructor_signature"],
        "gradient_scenarios": gradient_rows,
        "probability_gradient_scenarios": probability_rows,
    }

    with open(
        OUTPUT_DIR / "phase4_part28_diceceloss_foreground_gradient_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    lines = [
        "PHASE 4 - PART 28",
        "RSNA-ONLY DICECELOSS TARGET-ENCODING / FOREGROUND-GRADIENT AUDIT",
        "",
        "Evaluation only.",
        "No training performed.",
        "No model weights modified.",
        "No optimizer created.",
        "SPIDER not used.",
        "RSNA test set not used.",
        "",
        "TARGET ENCODING",
        json.dumps(shape_info, indent=2),
        "",
        "LOSS CONFIGURATION",
        json.dumps(loss_info["config"], indent=2),
        "",
        "GRADIENT SCENARIOS",
        json.dumps(gradient_rows, indent=2, default=str),
        "",
        "FOREGROUND/BACKGROUND GRADIENT COMPARISON",
        json.dumps(probability_rows, indent=2, default=str),
        "",
        "REAL LABEL CHECK",
        json.dumps(real_label_info, indent=2, default=str),
        "",
        "PART 11 DICE API",
        json.dumps(dice_api_info, indent=2, default=str),
    ]

    report_path = REPORT_DIR / "phase4_part28_diceceloss_foreground_gradient_audit_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {OUTPUT_DIR / 'part28_diceceloss_gradient_scenarios.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'part28_foreground_background_gradient_comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'phase4_part28_diceceloss_foreground_gradient_audit_summary.json'}")
    print(f"Saved: {report_path}")


def final_diagnosis(
    gradient_rows: List[Dict[str, Any]],
    probability_rows: List[Dict[str, Any]],
) -> str:
    valid = [
        r for r in gradient_rows
        if "error" not in r
        and r.get("finite_loss") is True
        and r.get("finite_grad") is True
    ]

    if not valid:
        return "LOSS_GRADIENT_AUDIT_INCONCLUSIVE"

    sparse = [
        r for r in valid
        if r.get("foreground_voxels", 0) > 0
    ]

    if not sparse:
        return "NO_FOREGROUND_SCENARIO_COMPLETED"

    finite_positive = [
        r for r in sparse
        if r.get("grad_abs_mean", 0) > 0
    ]

    if finite_positive:
        if probability_rows:
            return "DICECELOSS_PROVIDES_FOREGROUND_GRADIENT;_COLLAPSE_REQUIRES_TRAINING_PIPELINE_OR_CLASS_IMBALANCE_INVESTIGATION"
        return "DICECELOSS_FOREGROUND_GRADIENT_PRESENT"

    return "POSSIBLE_FOREGROUND_GRADIENT_FAILURE"


def main() -> None:
    banner("PHASE 4 - PART 28")
    print("RSNA-ONLY DICECELoss TARGET-ENCODING / FOREGROUND-GRADIENT AUDIT")
    print("")
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print("")
    print("Purpose:")
    print("Fix the Part 27 invalid gradient experiment and determine")
    print("whether DiceCELoss supplies a usable foreground learning signal.")

    print(f"\nPROJECT ROOT                         : {PROJECT_ROOT}")
    print(f"RSNA DATASET                         : {DATASET_ROOT}")
    print(f"PART 11 SOURCE                       : {PART11_SOURCE}")
    print(f"PART 15 CHECKPOINT                   : {PART15_CHECKPOINT}")
    print(f"OUTPUT DIRECTORY                     : {OUTPUT_DIR}")

    validate_paths()

    banner("PYTORCH / GPU ENVIRONMENT")
    print(f"PyTorch version                      : {torch.__version__}")
    print(f"CUDA available                       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device                               : {device}")
        print(f"GPU                                  : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory                            : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print(f"Device                               : {device}")

    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Feature size                          : {FEATURE_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")

    part11 = import_part11()

    loss_info = inspect_loss_configuration(part11)
    loss_fn = loss_info["loss"]

    shape_info = tensor_shapes()
    gradient_rows = run_gradient_scenarios(loss_fn)
    probability_rows = run_probability_gradient_scenarios(loss_fn)
    real_label_info = audit_real_target_encoding()
    dice_api_info = test_part11_dice_api_consistency(part11)

    diagnosis = final_diagnosis(
        gradient_rows,
        probability_rows,
    )

    banner("PART 28 FINAL SUMMARY")
    print(f"Completed gradient scenarios            : {len(gradient_rows)}")
    print(
        f"Successful finite scenarios             : "
        f"{sum(1 for x in gradient_rows if x.get('finite_grad') is True)}"
    )
    print(
        f"Foreground scenarios with nonzero grad  : "
        f"{sum(1 for x in gradient_rows if x.get('foreground_voxels', 0) > 0 and x.get('grad_abs_mean', 0) > 0)}"
    )
    print(f"Diagnosis                               : {diagnosis}")

    save_results(
        gradient_rows,
        probability_rows,
        shape_info,
        real_label_info,
        dice_api_info,
        loss_info,
    )

    banner("FINAL DECISION")
    if diagnosis.startswith("DICECELOSS_PROVIDES_FOREGROUND_GRADIENT"):
        print("PASS - DiceCELoss produces a finite foreground learning signal.")
        print("The Part 15 collapse should NOT be attributed to an absent loss gradient alone.")
        print("Next investigation should focus on the training pipeline, target construction")
        print("at patch level, class weighting/sampling, optimization dynamics, or checkpoint selection.")
    elif diagnosis == "POSSIBLE_FOREGROUND_GRADIENT_FAILURE":
        print("CAUTION - foreground gradient appears absent or negligible.")
        print("Investigate loss configuration and sparse-label handling before retraining.")
    else:
        print("INCONCLUSIVE - inspect the saved Part 28 JSON/report before making")
        print("a training decision.")

    print("")
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Model weights changed: NO")

    banner("PHASE 4 - PART 28 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner("PART 28 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
