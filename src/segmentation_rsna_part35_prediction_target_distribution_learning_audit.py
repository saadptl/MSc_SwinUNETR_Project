from __future__ import annotations
import json, math, random, sys, traceback
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
    from monai.losses import DiceCELoss
except Exception as exc:
    raise RuntimeError("MONAI is required for Part 35.") from exc

# ============================================================================
# PHASE 4 - PART 35
# RSNA-ONLY PREDICTION / TARGET DISTRIBUTION AND LEARNING-SIGNAL AUDIT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RSNA_ROOT = PROJECT_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
VAL_MANIFEST = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part8_dataset_construction" / "manifests" / "rsna_part8_validation_manifest.csv"
PART11_SOURCE = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
PART15_CHECKPOINT = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training" / "checkpoints" / "best_model.pth"
PART15_HISTORY = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training" / "part15_training_history.csv"
PART15_VAL_COHORT = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training" / "part15_validation_cohort.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part35_prediction_target_distribution_learning_audit"
CASE_CSV = OUTPUT_DIR / "part35_case_prediction_target_audit.csv"
CLASS_CSV = OUTPUT_DIR / "part35_class_distribution_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase4_part35_summary.json"
REPORT_TXT = OUTPUT_DIR / "phase4_part35_report.txt"

SEED = 42
VALIDATION_CASES = 100
NUM_CLASSES = 6
FEATURE_SIZE = 12
PATCH_SIZE = (64, 96, 96)
FOREGROUND_CLASSES = [1, 2, 3, 4, 5]
AMP_ENABLED = True

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def import_module(path, name):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def norm_id(v):
    if pd.isna(v):
        return ""
    try:
        return str(int(float(v)))
    except Exception:
        return str(v)

def get_case_ids(row):
    return norm_id(row.get("study_id")), norm_id(row.get("series_id"))

def load_part9(part11):
    if hasattr(part11, "load_part9_module"):
        return part11.load_part9_module()
    if hasattr(part11, "load_part9"):
        return part11.load_part9()
    path = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
    if not path.exists():
        raise RuntimeError("Could not obtain Part 9 loader.")
    return import_module(path, "part9_for_part35")

def ensure_image(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = torch.as_tensor(x).float()
    if x.ndim == 3:
        x = x[None, None]
    elif x.ndim == 4 and x.shape[0] == 1:
        x = x[None]
    elif x.ndim != 5:
        raise ValueError(f"Unexpected image shape: {tuple(x.shape)}")
    return x.contiguous()

def ensure_mask(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = torch.as_tensor(x).long()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {tuple(x.shape)}")
    return x.contiguous()

def resize_if_needed(image, mask):
    image = ensure_image(image)
    mask = ensure_mask(mask)
    if tuple(image.shape[-3:]) != PATCH_SIZE:
        image = F.interpolate(image, size=PATCH_SIZE, mode="trilinear", align_corners=False)
    if tuple(mask.shape) != PATCH_SIZE:
        mask = F.interpolate(mask[None, None].float(), size=PATCH_SIZE, mode="nearest").squeeze().long()
    return image, mask

def load_case(row, part11, part9):
    image, mask, info = part11.load_tensor_case(row, part9)

    if isinstance(image, torch.Tensor):
        if image.ndim == 5:
            image = image.squeeze(0).squeeze(0)
        elif image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    processed = part11.preprocess_case(image, mask)
    if isinstance(processed, tuple):
        image, mask = processed[0], processed[1]
    else:
        image = processed

    image, mask = resize_if_needed(image, mask)
    return image, mask, dict(info or {})

def create_model(part11, device):
    if hasattr(part11, "create_model"):
        return part11.create_model(device)
    model = SwinUNETR(in_channels=1, out_channels=NUM_CLASSES,
                      feature_size=FEATURE_SIZE, use_checkpoint=False)
    return model.to(device)

def load_checkpoint(model, device):
    ckpt = torch.load(PART15_CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return ckpt, missing, unexpected

def dice(tp, fp, fn):
    d = 2 * tp + fp + fn
    return float("nan") if d == 0 else 2.0 * tp / d

def audit(pred, target):
    pred, target = pred.long(), target.long()
    total = int(target.numel())
    tc = {c: int((target == c).sum()) for c in range(NUM_CLASSES)}
    pc = {c: int((pred == c).sum()) for c in range(NUM_CLASSES)}
    fg_t = sum(tc[c] for c in FOREGROUND_CLASSES)
    fg_p = sum(pc[c] for c in FOREGROUND_CLASSES)

    rows, ds = [], []
    for c in FOREGROUND_CLASSES:
        p, t = pred == c, target == c
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        d = dice(tp, fp, fn)
        if not math.isnan(d):
            ds.append(d)
        rows.append({
            "class_id": c, "class_name": CLASS_NAMES[c],
            "target_voxels": tc[c], "prediction_voxels": pc[c],
            "tp": tp, "fp": fp, "fn": fn, "dice": d,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "prediction_to_target_ratio": pc[c] / tc[c] if tc[c] else float("nan")
        })

    unique_p = sorted(int(x) for x in torch.unique(pred).cpu())
    unique_t = sorted(int(x) for x in torch.unique(target).cpu())
    labels = [c for c in FOREGROUND_CLASSES if pc[c] > 0]

    return {
        "total_voxels": total,
        "target_foreground_voxels": fg_t,
        "prediction_foreground_voxels": fg_p,
        "target_foreground_fraction": fg_t / total if total else 0.0,
        "prediction_foreground_fraction": fg_p / total if total else 0.0,
        "prediction_to_target_foreground_ratio": fg_p / fg_t if fg_t else float("nan"),
        "empty_prediction": fg_p == 0,
        "background_only_prediction": unique_p == [0],
        "single_foreground_class_prediction": len(labels) == 1,
        "predicted_foreground_labels": ",".join(map(str, labels)),
        "unique_prediction_labels": ",".join(map(str, unique_p)),
        "unique_target_labels": ",".join(map(str, unique_t)),
        "mean_foreground_dice": float(np.mean(ds)) if ds else float("nan"),
        "target_counts": tc, "prediction_counts": pc,
        "class_rows": rows,
    }

def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("PHASE 4 - PART 35")
    print("RSNA-ONLY PREDICTION / TARGET DISTRIBUTION AND LEARNING-SIGNAL AUDIT")
    print()
    print("Evaluation only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    section("PATH VALIDATION")
    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 8 validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
    }
    for name, p in paths.items():
        print(f"{name:<36}: {'FOUND' if p.exists() else 'MISSING'}")
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required Part 35 input(s):\n" + "\n".join(missing))

    section("PYTORCH / GPU ENVIRONMENT")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version                       : {torch.__version__}")
    print(f"CUDA available                        : {torch.cuda.is_available()}")
    print(f"Device                                : {device}")
    if device.type == "cuda":
        print(f"GPU                                   : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory                            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Feature size                          : {FEATURE_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")
    print(f"Validation cases                      : {VALIDATION_CASES}")

    section("IMPORTING VALIDATED PART 11")
    part11 = import_module(PART11_SOURCE, "part11_for_part35")
    print("✓ Corrected Part 11 imported.")

    import inspect
    for name in ("create_model", "preprocess_case", "load_tensor_case"):
        if hasattr(part11, name):
            print(f"{name:<20}: {inspect.signature(getattr(part11, name))}")

    section("LOADING PART 9 THROUGH PART 11")
    part9 = load_part9(part11)
    print("✓ Part 9 loader available.")

    section("RECONSTRUCTING EXACT VALIDATION COHORT")
    if hasattr(part11, "select_pilot_rows"):
        cohort = part11.select_pilot_rows(str(VAL_MANIFEST), VALIDATION_CASES, SEED)
        cohort = cohort if isinstance(cohort, pd.DataFrame) else pd.DataFrame(cohort)
        cohort = cohort.reset_index(drop=True)
        print("✓ Part 11 select_pilot_rows() used.")
    elif PART15_VAL_COHORT.exists():
        cohort = pd.read_csv(PART15_VAL_COHORT)
        print("✓ Saved Part 15 validation cohort used.")
    else:
        raise RuntimeError("No exact validation cohort API/file available.")
    print(f"Validation manifest rows               : {len(pd.read_csv(VAL_MANIFEST))}")
    print(f"Audit cohort rows                     : {len(cohort)}")
    if len(cohort) != VALIDATION_CASES:
        raise RuntimeError(f"Expected {VALIDATION_CASES} validation cases, got {len(cohort)}.")

    section("LOADING PART 15 HISTORY")
    history = pd.read_csv(PART15_HISTORY)
    best_history_dice = float(history["val_dice"].max()) if "val_dice" in history else float("nan")
    print(f"History rows                          : {len(history)}")
    print(f"Part 15 best recorded val Dice        : {best_history_dice:.6f}")

    section("LOADING PART 15 BEST CHECKPOINT")
    model = create_model(part11, device)
    ckpt, missing, unexpected = load_checkpoint(model, device)
    model.eval()
    print(f"Total parameters                      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Missing keys                          : {len(missing)}")
    print(f"Unexpected keys                       : {len(unexpected)}")
    if isinstance(ckpt, dict):
        print(f"Checkpoint epoch                      : {ckpt.get('epoch', 'N/A')}")
        print(f"Checkpoint best Dice                  : {ckpt.get('best_val_dice', 'N/A')}")

    loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

    section("RUNNING PART 35 PREDICTION / TARGET AUDIT")
    case_rows, class_rows, failed = [], [], []
    agg_t = {c: 0 for c in range(NUM_CLASSES)}
    agg_p = {c: 0 for c in range(NUM_CLASSES)}
    agg_tp = {c: 0 for c in FOREGROUND_CLASSES}
    agg_fp = {c: 0 for c in FOREGROUND_CLASSES}
    agg_fn = {c: 0 for c in FOREGROUND_CLASSES}

    with torch.no_grad():
        for i, row in cohort.iterrows():
            study, series = get_case_ids(row)
            try:
                image, mask, info = load_case(row, part11, part9)
                image, mask = image.to(device), mask.to(device)

                if device.type == "cuda" and AMP_ENABLED:
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        logits = model(image)
                        loss = loss_fn(logits, mask[None, None])
                else:
                    logits = model(image)
                    loss = loss_fn(logits, mask[None, None])

                pred = torch.argmax(logits, dim=1).squeeze(0)
                a = audit(pred, mask)

                for c in range(NUM_CLASSES):
                    agg_t[c] += a["target_counts"][c]
                    agg_p[c] += a["prediction_counts"][c]
                for r in a["class_rows"]:
                    c = r["class_id"]
                    agg_tp[c] += r["tp"]; agg_fp[c] += r["fp"]; agg_fn[c] += r["fn"]
                    class_rows.append({"case_index": i + 1, "study_id": study, "series_id": series, **r})

                case_rows.append({
                    "case_index": i + 1, "study_id": study, "series_id": series,
                    "status": "OK", "loss": float(loss.item()),
                    **{k: v for k, v in a.items() if k not in ("target_counts","prediction_counts","class_rows")},
                    **{f"target_class_{c}_voxels": a["target_counts"][c] for c in range(NUM_CLASSES)},
                    **{f"prediction_class_{c}_voxels": a["prediction_counts"][c] for c in range(NUM_CLASSES)},
                })

                if i < 5 or (i + 1) % 25 == 0 or i + 1 == len(cohort):
                    print(f"  [{i+1:03d}/{len(cohort)}] {study} | {series} | pred_fg={a['prediction_foreground_voxels']} | target_fg={a['target_foreground_voxels']} | labels=[{a['unique_prediction_labels']}] | Dice={a['mean_foreground_dice']:.6f}")
            except Exception as exc:
                failed.append({"case_index": i + 1, "study_id": study, "series_id": series, "status": "ERROR", "error": repr(exc)})
                case_rows.append(failed[-1])
                print(f"  [{i+1:03d}/{len(cohort)}] {study} | {series} | ERROR: {exc}")

    section("PART 35 OVERALL DIAGNOSIS")
    ok = len(case_rows) - len(failed)
    target_fg = sum(agg_t[c] for c in FOREGROUND_CLASSES)
    pred_fg = sum(agg_p[c] for c in FOREGROUND_CLASSES)
    ratio = pred_fg / target_fg if target_fg else float("nan")
    empty = sum(bool(r.get("empty_prediction", False)) for r in case_rows if r.get("status") == "OK")
    bg_only = sum(bool(r.get("background_only_prediction", False)) for r in case_rows if r.get("status") == "OK")
    single = sum(bool(r.get("single_foreground_class_prediction", False)) for r in case_rows if r.get("status") == "OK")
    present = [c for c in FOREGROUND_CLASSES if agg_p[c] > 0]

    class_summary = []
    ds = []
    for c in FOREGROUND_CLASSES:
        d = dice(agg_tp[c], agg_fp[c], agg_fn[c])
        if not math.isnan(d): ds.append(d)
        class_summary.append({
            "class_id": c, "class_name": CLASS_NAMES[c],
            "target_voxels": agg_t[c], "prediction_voxels": agg_p[c],
            "tp": agg_tp[c], "fp": agg_fp[c], "fn": agg_fn[c],
            "dice": d,
            "precision": agg_tp[c] / (agg_tp[c] + agg_fp[c]) if agg_tp[c] + agg_fp[c] else 0.0,
            "recall": agg_tp[c] / (agg_tp[c] + agg_fn[c]) if agg_tp[c] + agg_fn[c] else 0.0,
        })
    mean_dice = float(np.mean(ds)) if ds else float("nan")

    if pred_fg == 0:
        diagnosis = "COMPLETE_FOREGROUND_PREDICTION_COLLAPSE"
    elif len(present) == 1:
        diagnosis = "SINGLE_FOREGROUND_CLASS_COLLAPSE"
    elif ratio < 0.10:
        diagnosis = "SEVERE_FOREGROUND_UNDERPREDICTION"
    elif ratio < 0.50:
        diagnosis = "SUBSTANTIAL_FOREGROUND_UNDERPREDICTION"
    elif ratio > 2.0:
        diagnosis = "SEVERE_FOREGROUND_OVERPREDICTION"
    else:
        diagnosis = "FOREGROUND_PREDICTIONS_PRESENT"

    print(f"Successful cases                     : {ok}")
    print(f"Failed cases                         : {len(failed)}")
    print(f"Target foreground voxels             : {target_fg:,}")
    print(f"Predicted foreground voxels          : {pred_fg:,}")
    print(f"Prediction / target foreground ratio : {ratio:.6f}")
    print(f"Empty prediction cases               : {empty}")
    print(f"Background-only prediction cases     : {bg_only}")
    print(f"Single-foreground-class cases        : {single}")
    print(f"Predicted foreground classes         : {present}")
    print(f"Aggregate foreground Dice            : {mean_dice:.6f}")
    print(f"DIAGNOSIS                            : {diagnosis}")

    section("CLASS-WISE SUMMARY")
    class_df = pd.DataFrame(class_summary)
    print(class_df.to_string(index=False))

    section("PART 15 COMPARISON")
    checkpoint_dice = ckpt.get("best_val_dice") if isinstance(ckpt, dict) else None
    print(f"Part 15 history best val Dice        : {best_history_dice}")
    print(f"Checkpoint stored best Dice          : {checkpoint_dice}")
    print(f"Part 35 conventional Dice            : {mean_dice}")
    if checkpoint_dice is not None and not math.isnan(mean_dice):
        print(f"Checkpoint Dice - Part 35 Dice       : {float(checkpoint_dice) - mean_dice:.6f}")

    section("SAVING PART 35 OUTPUTS")
    pd.DataFrame(case_rows).to_csv(CASE_CSV, index=False)
    class_df.to_csv(CLASS_CSV, index=False)

    summary = {
        "phase": "Phase 4 - Part 35",
        "description": "RSNA-only prediction/target distribution and learning-signal audit",
        "validation_cases_requested": VALIDATION_CASES,
        "successful_cases": ok, "failed_cases": len(failed),
        "target_foreground_voxels": int(target_fg),
        "prediction_foreground_voxels": int(pred_fg),
        "prediction_target_foreground_ratio": float(ratio),
        "empty_prediction_cases": int(empty),
        "background_only_prediction_cases": int(bg_only),
        "single_foreground_class_prediction_cases": int(single),
        "predicted_foreground_classes": present,
        "aggregate_foreground_dice": float(mean_dice),
        "diagnosis": diagnosis,
        "part15_history_best_val_dice": float(best_history_dice),
        "checkpoint_best_val_dice": float(checkpoint_dice) if checkpoint_dice is not None else None,
        "class_summary": class_summary,
        "failed_cases_detail": failed,
        "patch_size": list(PATCH_SIZE), "feature_size": FEATURE_SIZE, "num_classes": NUM_CLASSES,
        "training_performed": False, "model_weights_modified": False,
        "optimizer_created": False, "optimizer_step_performed": False,
        "spider_used": False, "rsna_test_set_used": False,
        "pseudo_masks_are_manual_ground_truth": False,
    }
    if device.type == "cuda":
        summary["peak_gpu_allocated_gb"] = float(torch.cuda.max_memory_allocated(device) / 1024**3)
        summary["peak_gpu_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / 1024**3)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    REPORT_TXT.write_text(
        "PHASE 4 - PART 35\n"
        "RSNA-ONLY PREDICTION / TARGET DISTRIBUTION AND LEARNING-SIGNAL AUDIT\n"
        + "=" * 78 + "\n\n"
        + f"Successful cases: {ok}\nFailed cases: {len(failed)}\n"
        + f"Target foreground voxels: {target_fg}\nPredicted foreground voxels: {pred_fg}\n"
        + f"Prediction/target foreground ratio: {ratio}\n"
        + f"Empty predictions: {empty}\nBackground-only predictions: {bg_only}\n"
        + f"Single foreground class predictions: {single}\n"
        + f"Aggregate foreground Dice: {mean_dice}\nDiagnosis: {diagnosis}\n\n"
        + class_df.to_string(index=False) + "\n",
        encoding="utf-8",
    )

    print(f"Case audit CSV                        : {CASE_CSV}")
    print(f"Class summary CSV                     : {CLASS_CSV}")
    print(f"Summary JSON                          : {SUMMARY_JSON}")
    print(f"Report TXT                            : {REPORT_TXT}")

    section("PHASE 4 - PART 35 COMPLETE")
    print("✓ Prediction distribution calculated.")
    print("✓ Target distribution calculated.")
    print("✓ Per-class TP / FP / FN calculated.")
    print("✓ Conventional foreground Dice calculated.")
    print("✓ Prediction-collapse indicators calculated.")
    print("✓ Part 15 comparison recorded.")
    print("✓ No training performed.")
    print("✓ No model weights modified.")
    print("✓ No optimizer step performed.")
    print("✓ SPIDER not used.")
    print("✓ RSNA test set not used.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 78)
        print("PART 35 ERROR")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
