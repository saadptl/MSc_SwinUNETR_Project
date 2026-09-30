from __future__ import annotations

import csv
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


ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
PART11 = ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9 = ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"
P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
VACSV = P15 / "part15_validation_cohort.csv"
P58 = ROOT / "outputs" / "segmentation" / "rsna_part58_r2_lr_stability_confirmation"
OUT = ROOT / "outputs" / "segmentation" / "rsna_part59_probability_threshold_calibration_forensics"
REPORT = OUT / "reports"

VAL_N = 100
FULL = (64, 96, 96)
CROP = (32, 64, 64)
CLASSES = 6
RADIUS = 2
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
EPOCHS = range(1, 6)
CONDITIONS = {
    "A_constant_5e5": P58 / "A_constant_5e5",
    "B_step_1e4_to_5e5": P58 / "B_step_1e4_to_5e5",
}


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dilate6(x, n):
    x = x.astype(bool).copy()
    for _ in range(n):
        y = x.copy()
        y[1:] |= x[:-1]
        y[:-1] |= x[1:]
        y[:, 1:] |= x[:, :-1]
        y[:, :-1] |= x[:, 1:]
        y[:, :, 1:] |= x[:, :, :-1]
        y[:, :, :-1] |= x[:, :, 1:]
        x = y
    return x


def dilate_multiclass(mask):
    if RADIUS == 0:
        return mask.copy()
    regions = []
    for cid in range(1, CLASSES):
        r = mask == cid
        if r.any():
            regions.append((int(r.sum()), cid, dilate6(r, RADIUS)))
    regions.sort(key=lambda z: (-z[0], z[1]))
    out = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)
    for _, cid, r in regions:
        a = r & ~occupied
        out[a] = cid
        occupied |= a
    return out


def crop(image, mask):
    pts = np.argwhere(mask > 0)
    center = np.round(pts.mean(0)).astype(int) if pts.size else np.array([32, 48, 48])
    starts = [max(0, min(int(center[i]) - CROP[i] // 2, FULL[i] - CROP[i])) for i in range(3)]
    z, y, x = starts
    dz, dy, dx = CROP
    return image[z:z+dz, y:y+dy, x:x+dx].astype(np.float32), mask[z:z+dz, y:y+dy, x:x+dx].astype(np.int64)


def load_case(part11, part9, row):
    a = part11.load_tensor_case(row, part9)
    image, mask = a[0], a[1]
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    image, mask = np.asarray(image), np.asarray(mask)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if tuple(image.shape) != FULL or tuple(mask.shape) != FULL:
        raise RuntimeError(f"Unexpected shapes: {image.shape}, {mask.shape}")
    return crop(image, dilate_multiclass(mask.astype(np.int64)))


def preload(part11, part9, df):
    cases = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        image, mask = load_case(part11, part9, row)
        cases.append((image, mask))
        if i in (1, 25, 50, 75, 100):
            print(f"VALIDATION {i:03d}/{len(df)} FG={(mask > 0).sum()}")
    return cases


def threshold_predict(probs, threshold):
    fg = probs[1:]
    cls = np.argmax(fg, axis=0).astype(np.int64) + 1
    mx = np.max(fg, axis=0)
    cls[mx < threshold] = 0
    return cls


def binary(target, pred):
    t, p = target > 0, pred > 0
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    pc, tc = int(p.sum()), int(t.sum())
    den = pc + tc
    dice = 2 * tp / den if den else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dice, precision, recall, tp, fp, fn, pc, tc


def cdice(target, pred, cid):
    t, p = target == cid, pred == cid
    den = int(t.sum()) + int(p.sum())
    return 1.0 if den == 0 else float(2 * (t & p).sum() / den)


def evaluate(model, cases, device, threshold):
    model.eval()
    vals = []
    class_d = {str(c): [] for c in range(1, CLASSES)}
    class_p = {str(c): [] for c in range(1, CLASSES)}
    fg_probs, bg_probs = [], []

    for image, target in cases:
        x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, 1)[0].cpu().numpy()
        pred = threshold_predict(probs, threshold)
        d, pr, rc, tp, fp, fn, pc, tc = binary(target, pred)
        vals.append((d, pr, rc, tp, fp, fn, pc, tc))
        for c in range(1, CLASSES):
            class_d[str(c)].append(cdice(target, pred, c))
            class_p[str(c)].append(int((pred == c).sum()))
        fg_probs.append(float(probs[1:].sum(0).mean()))
        bg_probs.append(float(probs[0].mean()))
        del x, logits, probs, pred

    return {
        "threshold": threshold,
        "foreground_dice": float(np.mean([v[0] for v in vals])),
        "precision": float(np.mean([v[1] for v in vals])),
        "recall": float(np.mean([v[2] for v in vals])),
        "tp": int(sum(v[3] for v in vals)),
        "fp": int(sum(v[4] for v in vals)),
        "fn": int(sum(v[5] for v in vals)),
        "predicted_foreground_voxels": float(np.mean([v[6] for v in vals])),
        "target_foreground_voxels": float(np.mean([v[7] for v in vals])),
        "empty_cases": int(sum(v[6] == 0 for v in vals)),
        "total_cases": len(cases),
        "foreground_probability": float(np.mean(fg_probs)),
        "background_probability": float(np.mean(bg_probs)),
        "probability_gap": float(np.mean(fg_probs) - np.mean(bg_probs)),
        "per_class_dice": {k: float(np.mean(v)) for k, v in class_d.items()},
        "per_class_predicted_voxels": {k: float(np.mean(v)) for k, v in class_p.items()},
    }


def main():
    random.seed(159)
    np.random.seed(159)
    torch.manual_seed(159)

    print("=" * 82)
    print("PART 59 PATH VALIDATION")
    print("=" * 82)
    required = [("Project root", ROOT), ("Part 11", PART11), ("Part 9", PART9),
                ("Part 15 validation cohort", VACSV), ("Part 58 output", P58)]
    for name, path in required:
        print(f"{name:<40}: {'FOUND' if path.exists() else 'MISSING'}")
        if not path.exists():
            raise FileNotFoundError(path)
    for name, d in CONDITIONS.items():
        print(f"{name} directory".ljust(40) + f": {'FOUND' if d.exists() else 'MISSING'}")
        if not d.exists():
            raise FileNotFoundError(d)
        for e in EPOCHS:
            ck = d / "checkpoints" / f"epoch_{e:02d}.pth"
            if not ck.exists():
                raise FileNotFoundError(ck)

    print("\n" + "=" * 82)
    print("PART 59 — PROBABILITY THRESHOLD / CALIBRATION FORENSICS")
    print("=" * 82)
    print(f"Validation cohort : {VAL_N}")
    print("Pseudo-mask radius : R2")
    print(f"Full volume : {FULL}")
    print(f"Crop : {CROP}")
    print(f"Thresholds : {THRESHOLDS}")
    print("Checkpoints : epochs 1-5 from Part 58")
    print("Training performed : NO")
    print("Optimizer used : NO")
    print("Backward pass : NO")
    print("SPIDER : NO")
    print("Test set : NO")
    print("Part 15 modified : NO")

    part11 = load_mod(PART11, "part11_part59_runtime")
    part9 = load_mod(PART9, "part9_part59_runtime")
    df = pd.read_csv(VACSV).head(VAL_N)

    print("\n" + "=" * 82)
    print("PART 59 VALIDATION PRELOAD")
    print("=" * 82)
    cases = preload(part11, part9, df)

    print("\n" + "=" * 82)
    print("PART 59 SHAPE / LABEL SMOKE TEST")
    print("=" * 82)
    print(f"Image : {cases[0][0].shape}")
    print(f"Mask : {cases[0][1].shape}")
    print(f"Labels : {sorted(np.unique(cases[0][1]).tolist())}")
    assert cases[0][0].shape == CROP and cases[0][1].shape == CROP
    assert cases[0][1].min() >= 0 and cases[0][1].max() < CLASSES
    print("✓ Shape / label smoke test PASSED.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nPyTorch : {torch.__version__}")
    print(f"Device : {device}")

    records = []

    for condition, directory in CONDITIONS.items():
        print("\n" + "=" * 82)
        print(f"PART 59 CONDITION: {condition}")
        print("=" * 82)

        for epoch in EPOCHS:
            ckpt = directory / "checkpoints" / f"epoch_{epoch:02d}.pth"
            model = part11.create_model(device)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state_dict"], strict=True)

            print(f"\nCHECKPOINT EPOCH {epoch:02d} | SHA256={sha256(ckpt)}")

            for threshold in THRESHOLDS:
                m = evaluate(model, cases, device, threshold)
                print(
                    f"  T={threshold:.2f} | Dice={m['foreground_dice']:.6f} | "
                    f"P={m['precision']:.6f} | R={m['recall']:.6f} | "
                    f"PredFG={m['predicted_foreground_voxels']:.1f} | "
                    f"Empty={m['empty_cases']}/{VAL_N}"
                )
                row = {
                    "condition": condition, "epoch": epoch,
                    "checkpoint": str(ckpt), "threshold": threshold,
                    **{k: v for k, v in m.items() if k != "per_class_dice" and k != "per_class_predicted_voxels"},
                }
                for c in range(1, CLASSES):
                    row[f"class{c}_dice"] = m["per_class_dice"][str(c)]
                    row[f"class{c}_pred_voxels"] = m["per_class_predicted_voxels"][str(c)]
                records.append(row)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 82)
    print("PART 59 BEST THRESHOLD ANALYSIS")
    print("=" * 82)

    best_rows = []
    for condition in CONDITIONS:
        for epoch in EPOCHS:
            rows = [r for r in records if r["condition"] == condition and r["epoch"] == epoch]
            best = max(rows, key=lambda r: (r["foreground_dice"], r["recall"], -r["empty_cases"]))
            best_rows.append(best)
            print(
                f"{condition:<24} E{epoch} | T={best['threshold']:.2f} | "
                f"Dice={best['foreground_dice']:.6f} | P={best['precision']:.6f} | "
                f"R={best['recall']:.6f} | PredFG={best['predicted_foreground_voxels']:.1f} | "
                f"Empty={best['empty_cases']}/{VAL_N}"
            )

    global_best = max(best_rows, key=lambda r: (r["foreground_dice"], r["recall"], -r["empty_cases"]))
    argmax_rows = [r for r in records if abs(r["threshold"] - 0.50) < 1e-9]
    argmax_best = max(argmax_rows, key=lambda r: r["foreground_dice"])
    gain = global_best["foreground_dice"] - argmax_best["foreground_dice"]

    if gain > 0.01:
        diagnosis = "THRESHOLDING_REVEALS_SUBSTANTIAL_HIDDEN_FOREGROUND_SIGNAL"
    elif gain > 0.002:
        diagnosis = "THRESHOLDING_REVEALS_MODEST_HIDDEN_FOREGROUND_SIGNAL"
    else:
        diagnosis = "THRESHOLDING_DOES_NOT_RESCUE_FOREGROUND_PREDICTION"

    print("\n" + "=" * 82)
    print("PART 59 DIAGNOSTIC INTERPRETATION")
    print("=" * 82)
    print(f"Global best : {global_best['condition']} epoch {global_best['epoch']}")
    print(f"Best threshold : {global_best['threshold']:.2f}")
    print(f"Best thresholded FG Dice : {global_best['foreground_dice']:.6f}")
    print(f"Best T=0.50 FG Dice : {argmax_best['foreground_dice']:.6f}")
    print(f"Threshold gain : {gain:+.6f}")
    print(f"Diagnosis : {diagnosis}")

    REPORT.mkdir(parents=True, exist_ok=True)
    trajectory = REPORT / "part59_threshold_trajectory.csv"
    fields = list(records[0].keys())
    with open(trajectory, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    best_csv = REPORT / "part59_best_threshold_by_checkpoint.csv"
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        fields2 = list(best_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields2)
        w.writeheader()
        w.writerows(best_rows)

    summary = REPORT / "part59_summary.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({
            "part": 59,
            "purpose": "Threshold/calibration forensics using existing Part 58 R2 checkpoints.",
            "validation_cases": VAL_N,
            "pseudo_mask_radius": RADIUS,
            "thresholds": THRESHOLDS,
            "training_performed": False,
            "optimizer_used": False,
            "backward_pass": False,
            "conditions": list(CONDITIONS.keys()),
            "best_by_checkpoint": best_rows,
            "global_best": global_best,
            "standard_t50_best": argmax_best,
            "threshold_gain": gain,
            "diagnosis": diagnosis,
            "scientific_note": "Targets are pseudo-masks; this does not establish medical ground-truth segmentation accuracy."
        }, f, indent=2)

    print("\n" + "=" * 82)
    print("PART 59 COMPLETE")
    print("=" * 82)
    print(f"Output directory : {OUT}")
    print(f"Trajectory CSV : {trajectory}")
    print(f"Best-threshold CSV : {best_csv}")
    print(f"Summary JSON : {summary}")


if __name__ == "__main__":
    main()
