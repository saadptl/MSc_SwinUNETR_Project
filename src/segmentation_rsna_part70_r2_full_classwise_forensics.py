from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# PART 70 — R2_FULL CLASS-WISE SEGMENTATION FORENSICS
# ============================================================================
# Purpose:
#   Determine which of the five foreground classes are actually being learned
#   by the R2_FULL SwinUNETR model, using the already-trained Part 69
#   R2_FULL checkpoints. This is EVALUATION ONLY: no training, optimizer step,
#   checkpoint modification, SPIDER data, or test data.
#
# Locked evaluation:
#   - RSNA 2024 Lumbar Spine Degenerative Classification only
#   - first 50 Part 15 validation cases (same Part 69 validation cohort)
#   - Part 69 R2_FULL checkpoints: epoch 1, 2, 3 + initialization
#   - exact Part 69 preprocessing/crop geometry
#   - same Part 15 initialization SHA256
#   - SwinUNETR 3D, 1 input channel, 6 output classes, feature_size=12
#   - centered crop (32,64,64), center determined from R2_FULL foreground
#
# For every checkpoint and class, report:
#   - target voxel count
#   - predicted voxel count
#   - true-positive voxel count
#   - Dice
#   - precision
#   - recall
#   - F1 (same as Dice for binary class overlap)
#   - probability mean/max at target voxels
#   - probability mean/max overall
#   - prediction-empty rate
#
# IMPORTANT:
#   Per-class metrics are computed explicitly here rather than relying on
#   absent-class Dice conventions. This avoids misleading averages when a
#   class is not predicted.
# ============================================================================

ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC = ROOT / "src"
P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
VAL_CSV = P15 / "part15_validation_cohort.csv"
INIT_CKPT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"

P69 = ROOT / "outputs" / "segmentation" / "rsna_part69_core_halo_supervision_ablation"
CKPT_DIR = P69 / "checkpoints"
OUT = ROOT / "outputs" / "segmentation" / "rsna_part70_r2_full_classwise_forensics"
REPORT = OUT / "reports"

N_VAL = 50
FULL = (64, 96, 96)
CROP = (32, 64, 64)
BATCH_SIZE = 1
SEED = 42

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def banner(text: str):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_paths():
    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 validation cohort": VAL_CSV,
        "Part 15 initialization": INIT_CKPT,
        "Part 69 checkpoint directory": CKPT_DIR,
    }
    for name, path in required.items():
        ok = path.exists()
        print(f"{name:<34}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            raise FileNotFoundError(str(path))

    expected = [
        CKPT_DIR / "part69_r2_full_epoch1.pth",
        CKPT_DIR / "part69_r2_full_epoch2.pth",
        CKPT_DIR / "part69_r2_full_epoch3.pth",
    ]
    for path in expected:
        ok = path.exists()
        print(f"Required checkpoint {path.name:<20}: {'FOUND' if ok else 'MISSING'}")
        if not ok:
            raise FileNotFoundError(str(path))


def as_tensor_image(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float()
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image after squeeze, got {tuple(x.shape)}")
    return x


def as_mask(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D mask after squeeze, got {x.shape}")
    return x.astype(np.int64)


def centered_crop(arr: np.ndarray, center: np.ndarray):
    starts = []
    for dim, c, size in zip(arr.shape, center, CROP):
        start = int(round(float(c) - size / 2.0))
        start = max(0, min(start, dim - size))
        starts.append(start)
    z0, y0, x0 = starts
    crop = arr[z0:z0+CROP[0], y0:y0+CROP[1], x0:x0+CROP[2]]
    if crop.shape != CROP:
        raise ValueError(f"Crop shape {crop.shape} != {CROP}")
    return crop, tuple(starts)


def dilate_labels(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    if radius == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=np.int64)
    for c in range(1, 6):
        binary = mask == c
        if not binary.any():
            continue
        d = ndimage.binary_dilation(
            binary,
            iterations=radius,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        out[(d) & (out == 0)] = c
    out[mask > 0] = mask[mask > 0]
    return out


def make_r2_full(raw_mask: np.ndarray):
    resized = p11.resize_3d(raw_mask, FULL, is_mask=True)
    return dilate_labels(resized, 2)


class R2FullValidationDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        loaded = p11.load_tensor_case(row, p9)
        image = as_tensor_image(loaded[0])
        raw_mask = as_mask(loaded[1])
        r2 = make_r2_full(raw_mask)
        q = np.argwhere(r2 > 0)
        center = q.mean(axis=0) if len(q) else np.array([31.5, 47.5, 47.5])
        image_crop, starts = centered_crop(image.numpy(), center)
        target_crop, _ = centered_crop(r2, center)
        return {
            "image": torch.from_numpy(image_crop.astype(np.float32)).unsqueeze(0),
            "target": torch.from_numpy(target_crop.astype(np.int64)).unsqueeze(0),
            "case_no": int(idx + 1),
            "crop_z0": starts[0],
            "crop_y0": starts[1],
            "crop_x0": starts[2],
        }


def load_initial_state():
    obj = torch.load(INIT_CKPT, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Could not locate model state_dict in initialization checkpoint.")


def normalize_state_keys(state):
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def create_model():
    model = p11.create_model(DEVICE)
    state = normalize_state_keys(load_initial_state())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Initialization load mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.to(DEVICE)
    return model


def load_checkpoint_model(path: Path):
    model = create_model()
    obj = torch.load(path, map_location="cpu")
    state = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
    state = normalize_state_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch for {path.name}: missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.eval()
    return model


def safe_div(num, den):
    return float(num / den) if den else np.nan


def evaluate_checkpoint(model, loader, checkpoint_name):
    records = []
    case_records = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            tgt = target[:, 0]

            for b in range(image.shape[0]):
                pred_np = pred[b].detach().cpu().numpy()
                tgt_np = tgt[b].detach().cpu().numpy()
                prob_np = probs[b].detach().cpu().numpy()
                case_no = int(batch["case_no"][b])

                for c, class_name in CLASSES.items():
                    p = pred_np == c
                    t = tgt_np == c
                    tp = int((p & t).sum())
                    pred_count = int(p.sum())
                    target_count = int(t.sum())
                    fp = pred_count - tp
                    fn = target_count - tp
                    dice_den = pred_count + target_count
                    dice = safe_div(2.0 * tp, dice_den) if dice_den else 1.0
                    precision = safe_div(tp, pred_count)
                    recall = safe_div(tp, target_count)
                    union = pred_count + target_count - tp
                    iou = safe_div(tp, union)

                    target_probs = prob_np[c][t]
                    pred_probs = prob_np[c][p]
                    all_probs = prob_np[c].ravel()

                    records.append({
                        "checkpoint": checkpoint_name,
                        "case_no": case_no,
                        "class_id": c,
                        "class_name": class_name,
                        "target_voxels": target_count,
                        "predicted_voxels": pred_count,
                        "true_positive_voxels": tp,
                        "false_positive_voxels": fp,
                        "false_negative_voxels": fn,
                        "dice": dice,
                        "precision": precision,
                        "recall": recall,
                        "iou": iou,
                        "target_probability_mean": float(target_probs.mean()) if target_probs.size else np.nan,
                        "target_probability_max": float(target_probs.max()) if target_probs.size else np.nan,
                        "predicted_probability_mean": float(pred_probs.mean()) if pred_probs.size else np.nan,
                        "predicted_probability_max": float(pred_probs.max()) if pred_probs.size else np.nan,
                        "overall_probability_mean": float(all_probs.mean()),
                        "overall_probability_max": float(all_probs.max()),
                        "prediction_empty": int(pred_count == 0),
                    })

                case_records.append({
                    "checkpoint": checkpoint_name,
                    "case_no": case_no,
                    "total_target_fg": int((tgt_np > 0).sum()),
                    "total_pred_fg": int((pred_np > 0).sum()),
                    "all_foreground_prediction_empty": int((pred_np > 0).sum() == 0),
                })

            del logits, probs, pred, image, target

    return pd.DataFrame(records), pd.DataFrame(case_records)


def aggregate_class_metrics(case_df: pd.DataFrame):
    rows = []
    for (checkpoint, class_id, class_name), g in case_df.groupby(["checkpoint", "class_id", "class_name"], sort=False):
        # Micro counts across cases are the primary aggregate. Macro Dice is
        # retained separately to show case-to-case behavior.
        tp = int(g.true_positive_voxels.sum())
        pred_n = int(g.predicted_voxels.sum())
        target_n = int(g.target_voxels.sum())
        fp = int(g.false_positive_voxels.sum())
        fn = int(g.false_negative_voxels.sum())
        micro_dice = safe_div(2 * tp, pred_n + target_n)
        micro_precision = safe_div(tp, pred_n)
        micro_recall = safe_div(tp, target_n)
        micro_iou = safe_div(tp, pred_n + target_n - tp)
        macro_dice = float(g.dice.mean())
        macro_precision = float(g.precision.mean()) if g.precision.notna().any() else np.nan
        macro_recall = float(g.recall.mean()) if g.recall.notna().any() else np.nan
        rows.append({
            "checkpoint": checkpoint,
            "class_id": class_id,
            "class_name": class_name,
            "cases": len(g),
            "target_voxels_total": target_n,
            "predicted_voxels_total": pred_n,
            "true_positive_total": tp,
            "false_positive_total": fp,
            "false_negative_total": fn,
            "micro_dice": micro_dice,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_iou": micro_iou,
            "macro_case_dice": macro_dice,
            "macro_case_precision": macro_precision,
            "macro_case_recall": macro_recall,
            "empty_prediction_cases": int(g.prediction_empty.sum()),
            "empty_prediction_rate": float(g.prediction_empty.mean()),
            "mean_target_probability": float(g.target_probability_mean.mean()),
            "mean_max_target_probability": float(g.target_probability_max.mean()),
            "mean_overall_probability": float(g.overall_probability_mean.mean()),
            "mean_overall_probability_max": float(g.overall_probability_max.mean()),
        })
    return pd.DataFrame(rows)


def main():
    global p11, p9
    banner("PART 70 PATH VALIDATION")
    validate_paths()
    REPORT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    p11 = load_module(P11, "part11_part70")
    p9 = load_module(P9, "part9_part70")

    val_rows = pd.read_csv(VAL_CSV).head(N_VAL).copy()
    loader = DataLoader(
        R2FullValidationDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    init_hash = sha256_file(INIT_CKPT)
    checkpoint_paths = [
        ("INIT", None),
        ("E1", CKPT_DIR / "part69_r2_full_epoch1.pth"),
        ("E2", CKPT_DIR / "part69_r2_full_epoch2.pth"),
        ("E3", CKPT_DIR / "part69_r2_full_epoch3.pth"),
    ]

    banner("PART 70 LOCKED CLASS-WISE FORENSICS")
    print(f"Device                     : {DEVICE}")
    print(f"Validation cases           : {len(val_rows)}")
    print(f"Full shape                 : {FULL}")
    print(f"Centered crop              : {CROP}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(f"Seed                       : {SEED}")
    print(f"R2 supervision             : R2_FULL")
    print(f"Initialization SHA256      : {init_hash}")
    print("Evaluation checkpoints     : INIT / E1 / E2 / E3")
    print("Training performed in Part 70: NO")

    all_case = []
    all_class = []
    all_overall = []

    for name, path in checkpoint_paths:
        banner(f"PART 70 EVALUATION: {name}")
        if path is None:
            model = create_model()
        else:
            model = load_checkpoint_model(path)

        class_case_df, case_df = evaluate_checkpoint(model, loader, name)
        class_agg = aggregate_class_metrics(class_case_df)
        all_class.append(class_case_df)
        all_case.append(case_df)

        total_target = int(case_df.total_target_fg.sum())
        total_pred = int(case_df.total_pred_fg.sum())
        empty_cases = int(case_df.all_foreground_prediction_empty.sum())
        overall_tp = int(class_case_df.true_positive_voxels.sum())
        overall_dice = safe_div(2 * overall_tp, total_pred + total_target)

        # The overall foreground TP is the sum over mutually exclusive classes.
        overall_row = {
            "checkpoint": name,
            "target_foreground_voxels_total": total_target,
            "predicted_foreground_voxels_total": total_pred,
            "true_positive_foreground_voxels_total": int(class_case_df.true_positive_voxels.sum()),
            "foreground_dice_micro": overall_dice,
            "foreground_empty_prediction_cases": empty_cases,
            "foreground_empty_prediction_rate": empty_cases / len(case_df),
        }
        all_overall.append(overall_row)

        print(f"Overall foreground: target={total_target} pred={total_pred} Dice={overall_dice:.6f} empty={empty_cases}/{len(case_df)}")
        print("Class-wise micro Dice / precision / recall / predicted voxels:")
        for _, r in class_agg.iterrows():
            print(
                f"  C{int(r.class_id)} {r.class_name:<38} "
                f"Dice={r.micro_dice:.6f} "
                f"P={r.micro_precision:.6f} "
                f"R={r.micro_recall:.6f} "
                f"Pred={int(r.predicted_voxels_total):7d} "
                f"Target={int(r.target_voxels_total):7d} "
                f"Empty={int(r.empty_prediction_cases):2d}/{int(r.cases)}"
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    case_df = pd.concat(all_class, ignore_index=True)
    per_case_df = pd.concat(all_case, ignore_index=True)
    overall_df = pd.DataFrame(all_overall)
    class_df = aggregate_class_metrics(case_df)

    # Class trajectory table.
    class_df.to_csv(REPORT / "part70_classwise_metrics.csv", index=False)
    case_df.to_csv(REPORT / "part70_case_class_metrics.csv", index=False)
    per_case_df.to_csv(REPORT / "part70_case_foreground_summary.csv", index=False)
    overall_df.to_csv(REPORT / "part70_overall_foreground_trajectory.csv", index=False)

    # Wide table for direct comparison across checkpoints.
    wide = class_df.pivot(index=["class_id", "class_name"], columns="checkpoint", values="micro_dice").reset_index()
    wide.to_csv(REPORT / "part70_class_dice_trajectory.csv", index=False)

    # Determine the best class at the best overall checkpoint (E2 is expected
    # from Part 69, but we calculate it from the actual Part 70 evaluation).
    overall_best_row = overall_df.loc[overall_df.foreground_dice_micro.idxmax()]
    best_checkpoint = str(overall_best_row.checkpoint)
    best_class_df = class_df[class_df.checkpoint == best_checkpoint].copy()
    best_class_df = best_class_df.sort_values("micro_dice", ascending=False)

    class_summary = []
    for _, r in best_class_df.iterrows():
        class_summary.append({
            "class_id": int(r.class_id),
            "class_name": r.class_name,
            "micro_dice": float(r.micro_dice),
            "micro_precision": float(r.micro_precision) if pd.notna(r.micro_precision) else None,
            "micro_recall": float(r.micro_recall) if pd.notna(r.micro_recall) else None,
            "predicted_voxels_total": int(r.predicted_voxels_total),
            "target_voxels_total": int(r.target_voxels_total),
            "empty_prediction_rate": float(r.empty_prediction_rate),
        })

    # Conservative diagnosis focused on learning pattern rather than declaring
    # a medical conclusion. A class is considered effectively suppressed when
    # its micro Dice < 0.05 OR it is empty in >=80% of cases.
    suppressed = []
    learned = []
    for _, r in best_class_df.iterrows():
        if float(r.micro_dice) < 0.05 or float(r.empty_prediction_rate) >= 0.80:
            suppressed.append(int(r.class_id))
        if float(r.micro_dice) >= 0.20:
            learned.append(int(r.class_id))

    if len(suppressed) >= 4:
        diagnosis = "MULTI_CLASS_FOREGROUND_SUPPRESSION_PERSISTS"
    elif len(suppressed) > 0:
        diagnosis = "CLASS_SELECTIVE_FOREGROUND_SUPPRESSION"
    elif len(learned) >= 3:
        diagnosis = "MULTIPLE_FOREGROUND_CLASSES_SHOW_LEARNING_SIGNAL"
    else:
        diagnosis = "WEAK_MULTI_CLASS_FOREGROUND_LEARNING"

    summary = {
        "part": 70,
        "purpose": "R2_FULL class-wise segmentation forensics using Part 69 checkpoints",
        "evaluation_only": True,
        "training_performed": False,
        "validation_cases": len(val_rows),
        "full_shape": list(FULL),
        "crop_shape": list(CROP),
        "seed": SEED,
        "device": str(DEVICE),
        "initialization_checkpoint": str(INIT_CKPT),
        "initialization_sha256": init_hash,
        "evaluated_checkpoints": [x[0] for x in checkpoint_paths],
        "best_overall_checkpoint": best_checkpoint,
        "best_overall_foreground_dice": float(overall_best_row.foreground_dice_micro),
        "best_checkpoint_class_summary": class_summary,
        "suppressed_class_ids": suppressed,
        "classes_with_micro_dice_ge_0_20": learned,
        "diagnosis": diagnosis,
        "interpretation_limit": "RSNA coordinates are point annotations and Part 11 masks are pseudo-masks; these metrics evaluate model agreement with the pseudo-mask target, not medical ground-truth segmentation accuracy.",
    }
    (REPORT / "part70_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "PART 70 — R2_FULL CLASS-WISE SEGMENTATION FORENSICS",
        "",
        f"Device: {DEVICE}",
        f"Validation cases: {len(val_rows)}",
        f"Crop: {CROP}",
        f"Initialization SHA256: {init_hash}",
        "Evaluation only: YES",
        "",
        "OVERALL FOREGROUND TRAJECTORY",
    ]
    for _, r in overall_df.iterrows():
        report_lines.append(
            f"{r.checkpoint}: Dice={r.foreground_dice_micro:.6f}, "
            f"TargetFG={int(r.target_foreground_voxels_total)}, "
            f"PredFG={int(r.predicted_foreground_voxels_total)}, "
            f"Empty={int(r.foreground_empty_prediction_cases)}/{len(val_rows)}"
        )
    report_lines += ["", f"Best overall checkpoint: {best_checkpoint}", "", "BEST CHECKPOINT CLASS RANKING"]
    for _, r in best_class_df.iterrows():
        report_lines.append(
            f"C{int(r.class_id)} {r.class_name}: "
            f"Dice={r.micro_dice:.6f}, P={r.micro_precision:.6f}, R={r.micro_recall:.6f}, "
            f"Pred={int(r.predicted_voxels_total)}, Target={int(r.target_voxels_total)}, "
            f"Empty={int(r.empty_prediction_cases)}/{int(r.cases)}"
        )
    report_lines += ["", f"Diagnosis: {diagnosis}", "", summary["interpretation_limit"]]
    (REPORT / "part70_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    banner("PART 70 FINAL SUMMARY")
    print(f"Best overall checkpoint : {best_checkpoint}")
    print(f"Best overall FG Dice    : {float(overall_best_row.foreground_dice_micro):.6f}")
    print("Class-wise results at best checkpoint:")
    for _, r in best_class_df.iterrows():
        print(
            f"  C{int(r.class_id)} {r.class_name:<38} "
            f"Dice={r.micro_dice:.6f} P={r.micro_precision:.6f} R={r.micro_recall:.6f} "
            f"Pred={int(r.predicted_voxels_total)} Target={int(r.target_voxels_total)}"
        )
    print(f"Suppressed classes     : {suppressed if suppressed else 'NONE'}")
    print(f"Diagnosis               : {diagnosis}")
    print("\nReports:")
    print(REPORT / "part70_classwise_metrics.csv")
    print(REPORT / "part70_case_class_metrics.csv")
    print(REPORT / "part70_case_foreground_summary.csv")
    print(REPORT / "part70_overall_foreground_trajectory.csv")
    print(REPORT / "part70_class_dice_trajectory.csv")
    print(REPORT / "part70_summary.json")
    print(REPORT / "part70_report.txt")


if __name__ == "__main__":
    main()
