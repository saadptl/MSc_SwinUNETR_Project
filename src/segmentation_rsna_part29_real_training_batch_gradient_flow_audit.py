"""
PHASE 4 - PART 29
RSNA-ONLY REAL TRAINING-BATCH TARGET / INPUT / LOSS / GRADIENT FLOW AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer step is performed.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Part 28 proved that the configured DiceCELoss can produce a finite
foreground gradient. Part 29 therefore traces REAL Part 15 training
samples through the same data path and performs a single backward pass
only for gradient inspection.

IMPORTANT:
    backward() is performed on temporary cloned model parameters/tensors.
    The loaded Part 15 checkpoint is never modified and optimizer.step()
    is never called.

The audit checks:
  - exact Part 15 training cohort
  - real Part 9 -> Part 11 tensor loading
  - input shape/range/finite values
  - target shape/range/foreground occupancy
  - target class distribution
  - model output shape
  - raw logits/probabilities
  - DiceCELoss value
  - loss backward gradient
  - foreground/background output gradients
  - gradient flow into representative model parameters
  - whether the actual training batch has a usable foreground signal

This is NOT a training run.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RSNA_ROOT = (
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
PART15_TRAIN = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_train_cohort.csv"
)
PART15_VAL = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)
PART8_TRAIN = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_train_manifest.csv"
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
    / "rsna_part29_real_training_batch_gradient_flow_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
FEATURE_SIZE = 12
AUDIT_CASES = 10

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


def finite_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def validate_paths() -> None:
    banner("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 train cohort": PART15_TRAIN,
        "Part 15 validation cohort": PART15_VAL,
        "Part 8 train manifest": PART8_TRAIN,
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
            "Missing required Part 29 input(s):\n" + "\n".join(missing)
        )


def import_part11():
    banner("IMPORTING VALIDATED PART 11")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_part29",
        PART11_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not create Part 11 import specification.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    print("✓ Corrected Part 11 imported.")

    for name in (
        "create_model",
        "preprocess_case",
        "load_tensor_case",
        "dice_from_prediction",
    ):
        if not hasattr(module, name):
            raise AttributeError(f"Part 11 missing required API: {name}")
        print(f"{name:<22}: {inspect.signature(getattr(module, name))}")

    return module


def import_part9(part11):
    banner("LOADING PART 9")

    part9_path = (
        SRC_DIR
        / "segmentation_rsna_part9_3d_dataset_loader.py"
    )

    if not part9_path.exists():
        raise FileNotFoundError(
            f"Part 9 loader not found: {part9_path}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part9_part29",
        part9_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for {part9_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    print("✓ Part 9 loader imported.")
    print(f"Part 9 source : {part9_path}")

    return module

    raise RuntimeError(
        "Could not obtain Part 9 through Part 11 or a known Part 9 source."
    )


def load_checkpoint(part11, device):
    banner("LOADING PART 15 BEST CHECKPOINT")

    model = part11.create_model(device)
    model = model.to(device)

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
        if state is None:
            # Some checkpoints are themselves state dictionaries.
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state = checkpoint
            else:
                raise KeyError(
                    "Could not locate model state dictionary in checkpoint."
                )
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Total parameters : {sum(p.numel() for p in model.parameters())}")
    print(f"Missing keys     : {len(missing)}")
    print(f"Unexpected keys  : {len(unexpected)}")

    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    best_dice = checkpoint.get("best_dice") if isinstance(checkpoint, dict) else None

    print(f"Checkpoint epoch : {epoch}")
    print(f"Checkpoint best Dice : {best_dice}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    return model, checkpoint


def load_training_cohort() -> pd.DataFrame:
    banner("LOADING EXACT PART 15 TRAINING COHORT")

    df = pd.read_csv(PART15_TRAIN)

    print(f"Part 15 training cohort rows : {len(df)}")
    if len(df) == 0:
        raise ValueError("Part 15 training cohort is empty.")

    print(f"Columns : {list(df.columns)}")

    return df


def resolve_loader_call(part11, part9, row):
    """
    Call the exact Part 11 loader API.

    Signature established by Parts 25-28:
        load_tensor_case(row, part9)
    """
    return part11.load_tensor_case(row, part9)


def normalize_loaded_case(image, mask):
    """
    Normalize tensors to model-compatible dimensions without changing
    semantic content.

    Expected image:
        (C,D,H,W) or (1,C,D,H,W)

    Expected target:
        (D,H,W), (1,D,H,W), or (1,1,D,H,W)

    Returned:
        image  -> (1,C,D,H,W)
        target -> (1,1,D,H,W)
    """
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    image = image.detach().float()
    mask = mask.detach().long()

    if image.ndim == 4:
        image = image.unsqueeze(0)
    elif image.ndim == 5:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    if mask.ndim == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 4:
        # Could be (B,D,H,W) or (1,D,H,W).
        mask = mask.unsqueeze(1)
    elif mask.ndim == 5:
        pass
    else:
        raise ValueError(f"Unsupported target shape: {tuple(mask.shape)}")

    return image, mask


def target_statistics(target: torch.Tensor) -> Dict[str, Any]:
    t = target.detach().cpu().long()

    if t.ndim == 5:
        labels = t[:, 0]
    elif t.ndim == 4:
        labels = t
    else:
        raise ValueError(f"Unexpected target shape {tuple(t.shape)}")

    values, counts = torch.unique(labels, return_counts=True)

    distribution = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        distribution[str(int(value))] = int(count)

    total = int(labels.numel())
    foreground = int((labels > 0).sum().item())

    per_class = {}
    for c in range(NUM_CLASSES):
        n = int((labels == c).sum().item())
        per_class[str(c)] = {
            "class_name": CLASS_NAMES[c],
            "voxels": n,
            "fraction": n / total if total else 0.0,
        }

    return {
        "total_voxels": total,
        "foreground_voxels": foreground,
        "foreground_fraction": foreground / total if total else 0.0,
        "unique_labels": [int(v) for v in values.tolist()],
        "label_distribution": distribution,
        "per_class": per_class,
    }


def output_statistics(logits: torch.Tensor) -> Dict[str, Any]:
    probs = torch.softmax(logits.detach(), dim=1)
    prediction = torch.argmax(logits.detach(), dim=1)

    pred_values, pred_counts = torch.unique(
        prediction.cpu(),
        return_counts=True,
    )

    pred_distribution = {
        str(int(v)): int(c)
        for v, c in zip(pred_values.tolist(), pred_counts.tolist())
    }

    fg = int((prediction > 0).sum().item())
    total = int(prediction.numel())

    mean_probs = probs.mean(dim=(0, 2, 3, 4)).cpu()

    return {
        "logits_min": finite_float(logits.min().item()),
        "logits_max": finite_float(logits.max().item()),
        "logits_mean": finite_float(logits.mean().item()),
        "logits_std": finite_float(logits.std().item()),
        "probability_min": finite_float(probs.min().item()),
        "probability_max": finite_float(probs.max().item()),
        "mean_background_probability": finite_float(mean_probs[0].item()),
        "mean_foreground_probability": finite_float(
            mean_probs[1:].mean().item()
        ),
        "best_mean_foreground_probability": finite_float(
            mean_probs[1:].max().item()
        ),
        "argmax_distribution": pred_distribution,
        "foreground_prediction_voxels": fg,
        "foreground_prediction_fraction": fg / total if total else 0.0,
        "all_finite_logits": bool(torch.isfinite(logits).all()),
        "all_finite_probabilities": bool(torch.isfinite(probs).all()),
    }


def output_gradient_statistics(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
) -> Dict[str, Any]:
    """
    Backward on a temporary clone of logits.

    No model parameter is touched by this operation.
    """
    x = logits.detach().clone().requires_grad_(True)

    loss = loss_fn(x, target)
    if loss.ndim != 0:
        loss = loss.mean()

    loss.backward()

    grad = x.grad.detach()

    per_class = []
    for c in range(NUM_CLASSES):
        g = grad[:, c]
        per_class.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "abs_mean": finite_float(g.abs().mean().item()),
            "abs_max": finite_float(g.abs().max().item()),
            "l2": finite_float(torch.linalg.vector_norm(g).item()),
        })

    bg = per_class[0]["abs_mean"]
    fg_values = [x["abs_mean"] for x in per_class[1:]]
    fg_mean = float(np.mean(fg_values))

    return {
        "loss": finite_float(loss.item()),
        "gradient_abs_mean": finite_float(grad.abs().mean().item()),
        "gradient_abs_max": finite_float(grad.abs().max().item()),
        "gradient_l2": finite_float(torch.linalg.vector_norm(grad).item()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradient": bool(torch.isfinite(grad).all()),
        "background_logit_gradient_abs_mean": bg,
        "foreground_logit_gradient_abs_mean": fg_mean,
        "foreground_to_background_gradient_ratio": (
            fg_mean / bg if bg > 0 else float("nan")
        ),
        "per_class": per_class,
    }


def model_parameter_gradient_audit(
    model,
    image: torch.Tensor,
    target: torch.Tensor,
    loss_fn,
) -> Dict[str, Any]:
    """
    Real-model backward pass for diagnostic purposes only.

    The checkpoint is not updated. Gradients are cleared after inspection.
    """
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        logits = model(image)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss = loss_fn(logits, target)
        if loss.ndim != 0:
            loss = loss.mean()

        loss.backward()

    rows = []
    total_norm_sq = 0.0
    nonzero = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        g = param.grad.detach()
        norm = float(torch.linalg.vector_norm(g).item())
        mean_abs = float(g.abs().mean().item())

        total_norm_sq += norm * norm

        if norm > 0:
            nonzero += 1

        rows.append({
            "parameter": name,
            "numel": int(param.numel()),
            "grad_l2": norm,
            "grad_abs_mean": mean_abs,
            "grad_abs_max": float(g.abs().max().item()),
            "finite": bool(torch.isfinite(g).all()),
        })

    total_norm = math.sqrt(total_norm_sq)

    # Remove gradients immediately. No optimizer step occurs.
    model.zero_grad(set_to_none=True)

    return {
        "loss": finite_float(loss.item()),
        "parameter_tensors_with_grad": len(rows),
        "parameter_tensors_with_nonzero_grad": nonzero,
        "total_parameter_gradient_l2": total_norm,
        "parameters": rows,
    }


def audit_case(
    model,
    part11,
    part9,
    row,
    loss_fn,
    device,
    run_parameter_backward: bool,
) -> Dict[str, Any]:
    image, mask, info = resolve_loader_call(
        part11,
        part9,
        row,
    )

    image, target = normalize_loaded_case(image, mask)

    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    result = {
        "study_id": str(row.get("study_id", "")),
        "series_id": str(row.get("series_id", "")),
        "image_shape": list(image.shape),
        "target_shape": list(target.shape),
        "image_dtype": str(image.dtype),
        "target_dtype": str(target.dtype),
        "image_min": finite_float(image.min().item()),
        "image_max": finite_float(image.max().item()),
        "image_mean": finite_float(image.mean().item()),
        "image_std": finite_float(image.std().item()),
        "image_finite": bool(torch.isfinite(image).all()),
        "target_stats": target_statistics(target),
        "loader_info": {
            k: str(v)
            for k, v in (info or {}).items()
            if isinstance(k, str)
        },
    }

    # Model forward. No gradient needed for this first inspection.
    model.eval()
    with torch.no_grad():
        logits = model(image)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

    result["logits_shape"] = list(logits.shape)
    result["output_stats"] = output_statistics(logits)

    # Temporary output-gradient audit.
    result["output_gradient"] = output_gradient_statistics(
        logits,
        target,
        loss_fn,
    )

    # Optional actual model backward. Still no optimizer step.
    if run_parameter_backward:
        result["parameter_gradient"] = model_parameter_gradient_audit(
            model,
            image,
            target,
            loss_fn,
        )
    else:
        result["parameter_gradient"] = {
            "status": "SKIPPED",
        }

    return result


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in results if "error" not in r]

    if not successful:
        return {
            "cases_attempted": len(results),
            "cases_successful": 0,
        }

    fg_targets = [
        r["target_stats"]["foreground_voxels"]
        for r in successful
    ]
    fg_predictions = [
        r["output_stats"]["foreground_prediction_voxels"]
        for r in successful
    ]
    losses = [
        r["output_gradient"]["loss"]
        for r in successful
    ]
    output_gradients = [
        r["output_gradient"]["gradient_abs_mean"]
        for r in successful
    ]
    fg_gradients = [
        r["output_gradient"]["foreground_logit_gradient_abs_mean"]
        for r in successful
    ]
    bg_gradients = [
        r["output_gradient"]["background_logit_gradient_abs_mean"]
        for r in successful
    ]

    return {
        "cases_attempted": len(results),
        "cases_successful": len(successful),
        "mean_target_foreground_voxels": float(np.mean(fg_targets)),
        "min_target_foreground_voxels": int(np.min(fg_targets)),
        "max_target_foreground_voxels": int(np.max(fg_targets)),
        "mean_prediction_foreground_voxels": float(np.mean(fg_predictions)),
        "empty_prediction_cases": int(sum(x == 0 for x in fg_predictions)),
        "mean_loss": float(np.mean(losses)),
        "mean_output_gradient_abs": float(np.mean(output_gradients)),
        "mean_foreground_logit_gradient_abs": float(np.mean(fg_gradients)),
        "mean_background_logit_gradient_abs": float(np.mean(bg_gradients)),
        "mean_foreground_background_gradient_ratio": (
            float(np.mean([
                x / y for x, y in zip(fg_gradients, bg_gradients)
                if y > 0
            ]))
            if any(y > 0 for y in bg_gradients)
            else float("nan")
        ),
        "all_input_finite": all(
            r["image_finite"] for r in successful
        ),
        "all_logits_finite": all(
            r["output_stats"]["all_finite_logits"]
            for r in successful
        ),
        "all_gradients_finite": all(
            r["output_gradient"]["finite_gradient"]
            for r in successful
        ),
    }


def save_results(results, aggregate_result):
    banner("SAVING PART 29 RESULTS")

    case_rows = []

    for r in results:
        if "error" in r:
            case_rows.append({
                "study_id": r.get("study_id", ""),
                "series_id": r.get("series_id", ""),
                "error": r["error"],
            })
            continue

        ts = r["target_stats"]
        os = r["output_stats"]
        gs = r["output_gradient"]

        case_rows.append({
            "study_id": r["study_id"],
            "series_id": r["series_id"],
            "image_shape": str(r["image_shape"]),
            "target_shape": str(r["target_shape"]),
            "image_min": r["image_min"],
            "image_max": r["image_max"],
            "image_mean": r["image_mean"],
            "image_std": r["image_std"],
            "target_foreground_voxels": ts["foreground_voxels"],
            "target_foreground_fraction": ts["foreground_fraction"],
            "target_unique_labels": str(ts["unique_labels"]),
            "pred_foreground_voxels": os["foreground_prediction_voxels"],
            "pred_foreground_fraction": os["foreground_prediction_fraction"],
            "mean_background_probability": os["mean_background_probability"],
            "mean_foreground_probability": os["mean_foreground_probability"],
            "best_mean_foreground_probability": os["best_mean_foreground_probability"],
            "logits_min": os["logits_min"],
            "logits_max": os["logits_max"],
            "logits_mean": os["logits_mean"],
            "logits_std": os["logits_std"],
            "loss": gs["loss"],
            "output_gradient_abs_mean": gs["gradient_abs_mean"],
            "background_gradient_abs_mean": gs[
                "background_logit_gradient_abs_mean"
            ],
            "foreground_gradient_abs_mean": gs[
                "foreground_logit_gradient_abs_mean"
            ],
            "foreground_background_gradient_ratio": gs[
                "foreground_to_background_gradient_ratio"
            ],
            "finite_gradient": gs["finite_gradient"],
        })

    pd.DataFrame(case_rows).to_csv(
        OUTPUT_DIR / "part29_real_training_batch_flow_case_metrics.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "phase4_part29_real_training_batch_gradient_flow_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "phase": "PHASE 4 - PART 29",
                "evaluation_only": True,
                "training_performed": False,
                "optimizer_step_performed": False,
                "model_weights_modified": False,
                "spider_used": False,
                "rsna_test_set_used": False,
                "aggregate": aggregate_result,
                "cases": results,
            },
            f,
            indent=2,
            default=str,
        )

    report = [
        "PHASE 4 - PART 29",
        "RSNA-ONLY REAL TRAINING-BATCH TARGET / INPUT / LOSS / GRADIENT FLOW AUDIT",
        "",
        "Evaluation only.",
        "No training performed.",
        "No optimizer step performed.",
        "No model weights modified.",
        "SPIDER not used.",
        "RSNA test set not used.",
        "",
        "AGGREGATE RESULT",
        json.dumps(aggregate_result, indent=2, default=str),
        "",
        "CASE RESULTS",
        json.dumps(results, indent=2, default=str),
    ]

    report_path = REPORT_DIR / "phase4_part29_real_training_batch_gradient_flow_audit_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(
        f"Saved: {OUTPUT_DIR / 'part29_real_training_batch_flow_case_metrics.csv'}"
    )
    print(
        f"Saved: {OUTPUT_DIR / 'phase4_part29_real_training_batch_gradient_flow_audit_summary.json'}"
    )
    print(f"Saved: {report_path}")


def main():
    banner("PHASE 4 - PART 29")
    print("RSNA-ONLY REAL TRAINING-BATCH TARGET / INPUT / LOSS / GRADIENT FLOW AUDIT")
    print("")
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print("")
    print("Part 29 traces real Part 15 training samples through the")
    print("validated Part 9 -> Part 11 data path and measures whether")
    print("the actual training tensors provide a usable foreground signal.")

    print(f"\nPROJECT ROOT                         : {PROJECT_ROOT}")
    print(f"RSNA DATASET                         : {RSNA_ROOT}")
    print(f"PART 11 SOURCE                       : {PART11_SOURCE}")
    print(f"PART 15 CHECKPOINT                   : {PART15_CHECKPOINT}")
    print(f"PART 15 TRAIN COHORT                 : {PART15_TRAIN}")
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
    print(f"Audit cases                           : {AUDIT_CASES}")

    part11 = import_part11()
    part9 = import_part9(part11)

    train_df = load_training_cohort()

    model, checkpoint = load_checkpoint(
        part11,
        device,
    )

    from monai.losses import DiceCELoss

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
    )

    banner("LOSS CONFIGURATION")
    print("Loss                                  : DiceCELoss")
    print("to_onehot_y                           : True")
    print("softmax                               : True")
    print("Optimizer                             : NONE")
    print("Optimizer step                        : NONE")

    audit_df = train_df.head(min(AUDIT_CASES, len(train_df))).copy()

    banner("STARTING REAL TRAINING-BATCH FLOW AUDIT")
    print(f"Cases selected                         : {len(audit_df)}")

    results = []

    for idx, (_, row) in enumerate(audit_df.iterrows(), start=1):
        print(
            f"\n[{idx:03d}/{len(audit_df):03d}] "
            f"study={row.get('study_id', '')} "
            f"series={row.get('series_id', '')}"
        )

        try:
            # Perform a true model-parameter backward pass only for the
            # first successful real training sample. This reveals whether
            # gradients reach the network while still guaranteeing that
            # no optimizer step occurs.
            run_parameter_backward = idx == 1

            result = audit_case(
                model=model,
                part11=part11,
                part9=part9,
                row=row,
                loss_fn=loss_fn,
                device=device,
                run_parameter_backward=run_parameter_backward,
            )

            results.append(result)

            ts = result["target_stats"]
            os = result["output_stats"]
            gs = result["output_gradient"]

            print(
                f"  image={result['image_shape']} "
                f"target={result['target_shape']}"
            )
            print(
                f"  target_fg={ts['foreground_voxels']} "
                f"({ts['foreground_fraction']:.8f}) "
                f"labels={ts['unique_labels']}"
            )
            print(
                f"  pred_fg={os['foreground_prediction_voxels']} "
                f"bg_prob={os['mean_background_probability']:.6f} "
                f"best_fg_prob={os['best_mean_foreground_probability']:.6f}"
            )
            print(
                f"  loss={gs['loss']:.8f} "
                f"grad={gs['gradient_abs_mean']:.8e} "
                f"fg/bg_grad={gs['foreground_to_background_gradient_ratio']:.6f}"
            )

            if run_parameter_backward:
                pg = result["parameter_gradient"]
                print(
                    f"  parameter tensors with grad="
                    f"{pg['parameter_tensors_with_grad']} "
                    f"nonzero={pg['parameter_tensors_with_nonzero_grad']} "
                    f"total_grad_l2={pg['total_parameter_gradient_l2']:.8e}"
                )

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            results.append({
                "study_id": str(row.get("study_id", "")),
                "series_id": str(row.get("series_id", "")),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    aggregate_result = aggregate(results)

    banner("PART 29 REAL TRAINING-BATCH FLOW SUMMARY")
    print(f"Cases attempted                       : {aggregate_result['cases_attempted']}")
    print(f"Cases successful                      : {aggregate_result['cases_successful']}")

    if aggregate_result["cases_successful"]:
        print(
            f"Mean target foreground voxels       : "
            f"{aggregate_result['mean_target_foreground_voxels']:.4f}"
        )
        print(
            f"Mean predicted foreground voxels    : "
            f"{aggregate_result['mean_prediction_foreground_voxels']:.4f}"
        )
        print(
            f"Empty prediction cases               : "
            f"{aggregate_result['empty_prediction_cases']}"
        )
        print(
            f"Mean loss                            : "
            f"{aggregate_result['mean_loss']:.8f}"
        )
        print(
            f"Mean output gradient                 : "
            f"{aggregate_result['mean_output_gradient_abs']:.8e}"
        )
        print(
            f"Mean foreground logit gradient      : "
            f"{aggregate_result['mean_foreground_logit_gradient_abs']:.8e}"
        )
        print(
            f"Mean background logit gradient      : "
            f"{aggregate_result['mean_background_logit_gradient_abs']:.8e}"
        )
        print(
            f"Mean FG/BG gradient ratio            : "
            f"{aggregate_result['mean_foreground_background_gradient_ratio']:.6f}"
        )

    save_results(
        results,
        aggregate_result,
    )

    banner("FINAL DECISION")

    successful = [
        r for r in results
        if "error" not in r
    ]

    if not successful:
        print("INCONCLUSIVE - no real training sample completed the flow audit.")
    elif (
        aggregate_result["all_input_finite"]
        and aggregate_result["all_logits_finite"]
        and aggregate_result["all_gradients_finite"]
        and aggregate_result["mean_target_foreground_voxels"] > 0
        and aggregate_result["mean_foreground_logit_gradient_abs"] > 0
    ):
        if aggregate_result["empty_prediction_cases"] == aggregate_result["cases_successful"]:
            print("CRITICAL FINDING:")
            print("Real training labels contain foreground and real model outputs")
            print("produce a finite foreground gradient, but the checkpoint predicts")
            print("background-only on the audited training samples.")
            print("")
            print("This shifts the investigation toward TRAINING DYNAMICS /")
            print("DATA-SAMPLING / PATCH-COVERAGE / CHECKPOINT-SELECTION rather")
            print("than a fundamentally broken DiceCELoss implementation.")
        else:
            print("PASS:")
            print("Real training samples contain labels and produce a finite")
            print("foreground learning signal through the complete audit path.")
    else:
        print("CAUTION / INCONCLUSIVE:")
        print("One or more real training tensors failed the expected integrity checks.")

    print("")
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Optimizer step       : NO")
    print("Model weights changed: NO")

    banner("PHASE 4 - PART 29 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner("PART 29 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
