"""
PART 75.1 — CORRECTED
RSNA-only C2/C3 supervision + anatomical localization audit.

Correction:
The Part 75 annotation section is replaced with the exact Part 67 /
Part 68.1 RSNA train_label_coordinates.csv mapping:
study_id + series_id -> instance_number -> native z -> FULL coordinates,
followed by the exact foreground-centered R2 crop and rounding handling.

Forensic only: no training, optimizer, backward pass, SPIDER, or test set.
RSNA coordinates are point/localizer annotations, not segmentation masks.
"""

from pathlib import Path
import importlib.util
import json
import gc
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC = ROOT / "src"
P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"
P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
VAL_CSV = P15 / "part15_validation_cohort.csv"
RSNA = ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
COORD_CSV = RSNA / "train_label_coordinates.csv"
CKPT = ROOT / "outputs" / "segmentation" / "rsna_part71_class_balanced_dicece_training" / "checkpoints" / "part71_r2_full_class_balanced_epoch3.pth"
REPORT = ROOT / "outputs" / "segmentation" / "rsna_part75_1_c2_c3_supervision_localization_audit_corrected" / "reports"
REPORT.mkdir(parents=True, exist_ok=True)

VAL_N = 50
FULL = (64, 96, 96)
CROP = (32, 64, 64)
FOCUS = [2, 3]
THRESH = {2: 0.15, 3: 0.20}
CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}
NAME = {v: k for k, v in CLASSES.items()}
NAME.update({
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
})
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def div(a, b):
    return float(a / b) if b else 0.0


def dilate_labels(mask, radius):
    mask = np.asarray(mask, dtype=np.int64)
    if radius == 0:
        return mask.copy()
    out = mask.copy()
    for c in range(1, 6):
        q = mask == c
        if not q.any():
            continue
        expanded = ndimage.binary_dilation(
            q, structure=np.ones((2 * radius + 1,) * 3, dtype=bool)
        )
        out[expanded & (out == 0)] = c
    return out


def map_coord(x, y, z, native):
    nz, nh, nw = map(float, native)
    return (
        (z + 0.5) * FULL[0] / nz - 0.5,
        (y + 0.5) * FULL[1] / nh - 0.5,
        (x + 0.5) * FULL[2] / nw - 0.5,
    )


def centered_crop(mask, center):
    starts = [
        max(0, min(int(round(float(center[i]))) - CROP[i] // 2,
                FULL[i] - CROP[i]))
        for i in range(3)
    ]
    z, y, x = starts
    dz, dy, dx = CROP
    return mask[z:z+dz, y:y+dy, x:x+dx], tuple(starts)


def exact_points(row, coord, p11, native):
    sid = str(row["study_id"])
    ser = str(row["series_id"])
    ann = coord[
        (coord.study_id.astype(str) == sid) &
        (coord.series_id.astype(str) == ser)
    ]

    series_dir = p11.resolve_series_dir(row)
    if series_dir is None:
        return []

    _, ds, _ = p11.read_dicom_series_robust(series_dir)
    inst_to_z = {
        int(d.get("InstanceNumber", 0)): i
        for i, d in enumerate(ds)
    }

    points = []
    for _, a in ann.iterrows():
        cid = NAME.get(str(a["condition"]).strip())
        try:
            x = float(a["x"])
            y = float(a["y"])
            inst = int(float(a["instance_number"]))
        except Exception:
            continue

        if cid is None or inst not in inst_to_z:
            continue
        z = inst_to_z[inst]

        if not (0 <= x < native[2] and 0 <= y < native[1] and 0 <= z < native[0]):
            continue

        zz, yy, xx = map_coord(x, y, z, native)
        points.append({
            "class_id": cid,
            "class_name": CLASSES[cid],
            "z": zz, "y": yy, "x": xx,
            "condition": str(a["condition"]),
            "level": str(a.get("level", "")),
            "instance_number": inst,
            "native_x": x, "native_y": y, "native_z": z,
        })
    return points


def load_case(row, p9, p11, coord):
    im, m, _ = p11.load_case_robust(row, p9)
    im = np.asarray(im, np.float32)
    m = np.asarray(m, np.int64)
    native = tuple(int(v) for v in m.shape)

    im_full = p11.resize_3d(im, FULL, is_mask=False)
    m_full = p11.resize_3d(m, FULL, is_mask=True)
    r2 = dilate_labels(m_full, 2)

    q = np.argwhere(r2 > 0)
    center = q.mean(axis=0) if len(q) else np.array([31.5, 47.5, 47.5])
    _, starts = centered_crop(r2, center)

    z, y, x = starts
    dz, dy, dx = CROP
    im_crop = im_full[z:z+dz, y:y+dy, x:x+dx]
    target = r2[z:z+dz, y:y+dy, x:x+dx]

    points = exact_points(row, coord, p11, native)
    return torch.tensor(im_crop, dtype=torch.float32), torch.tensor(target, dtype=torch.long), starts, points


def inside(p):
    return bool(all(0 <= p[i] < CROP[i] for i in range(3)))


def nearest(point, mask):
    if not mask.any():
        return float("inf")
    p = np.round(point).astype(int)
    p = np.clip(p, 0, np.asarray(CROP) - 1)
    if mask[tuple(p)]:
        return 0.0
    return float(ndimage.distance_transform_edt(~mask)[tuple(p)])


def mask_dist(source, target):
    if not source.any() or not target.any():
        return float("inf")
    return float(ndimage.distance_transform_edt(~target)[source].mean())


def top_mask(prob, fraction=0.01):
    flat = prob.ravel()
    n = max(1, int(round(flat.size * fraction)))
    idx = np.argpartition(flat, -n)[-n:]
    out = np.zeros_like(flat, dtype=bool)
    out[idx] = True
    return out.reshape(prob.shape)


def main():
    print("=" * 88)
    print("PART 75.1 — CORRECTED C2/C3 SUPERVISION + LOCALIZATION AUDIT")
    print("=" * 88)
    print("Device              :", DEVICE)
    print("Validation cases    :", VAL_N)
    print("Full / crop         :", FULL, "/", CROP)
    print("Checkpoint          :", CKPT)
    print("Annotation source   :", COORD_CSV)
    print("Annotation pipeline : EXACT Part 67 / Part 68.1")
    print("Training            : NO")
    print("Optimizer           : NO")
    print("Backward pass       : NO")
    print("SPIDER              : NO")
    print("RSNA test set       : NO")

    if not VAL_CSV.exists():
        raise FileNotFoundError(VAL_CSV)
    if not COORD_CSV.exists():
        raise FileNotFoundError(COORD_CSV)
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    p11 = loadmod(P11, "p11_part75_1")
    p9 = loadmod(P9, "p9_part75_1")
    val = pd.read_csv(VAL_CSV).head(VAL_N)
    coord = pd.read_csv(COORD_CSV)

    required = {"study_id", "series_id", "instance_number", "condition", "level", "x", "y"}
    missing = required - set(coord.columns)
    if missing:
        raise RuntimeError(f"Missing RSNA coordinate columns: {sorted(missing)}")

    ck = torch.load(CKPT, map_location="cpu")
    state = ck.get("model_state_dict", ck)
    model = p11.create_model(DEVICE).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    records = []
    ann_records = []
    total_points = 0
    ann_cases = 0

    with torch.no_grad():
        for case_no, (_, row) in enumerate(val.iterrows(), 1):
            image, target, starts, points = load_case(row, p9, p11, coord)
            total_points += len(points)
            ann_cases += int(bool(points))

            x = image.unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                logits = model(x)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            target_np = target.numpy()

            if case_no in (1, 10, 20, 25, 40, 50):
                print(f"VALIDATION {case_no:03d}/{VAL_N} R2_FG={(target_np > 0).sum():5d} points={len(points):2d}")

            for c in FOCUS:
                prob = probs[c]
                truth = target_np == c
                pred = prob >= THRESH[c]
                top1 = top_mask(prob, 0.01)
                tn = int(truth.sum())
                pn = int(pred.sum())
                tp = int((pred & truth).sum())

                cp = [p for p in points if p["class_id"] == c]
                inside_points = []
                for p in cp:
                    local = np.array([p["z"] - starts[0], p["y"] - starts[1], p["x"] - starts[2]], dtype=float)
                    if inside(local):
                        inside_points.append((p, local))

                records.append({
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "target_voxels": tn,
                    "predicted_voxels": pn,
                    "prediction_target_ratio": div(pn, tn),
                    "dice": div(2 * tp, pn + tn),
                    "precision": div(tp, pn),
                    "recall": div(tp, tn),
                    "prediction_to_target_distance": mask_dist(pred, truth),
                    "target_to_prediction_distance": mask_dist(truth, pred),
                    "top1_to_target_distance": mask_dist(top1, truth),
                    "target_to_top1_distance": mask_dist(truth, top1),
                    "annotation_count": len(cp),
                    "annotation_inside_crop": len(inside_points),
                    "annotation_to_target_distance":
                        float(np.mean([nearest(q, truth) for _, q in inside_points])) if inside_points else -1.0,
                    "annotation_to_prediction_distance":
                        float(np.mean([nearest(q, pred) for _, q in inside_points])) if inside_points else -1.0,
                    "annotation_to_top1_distance":
                        float(np.mean([nearest(q, top1) for _, q in inside_points])) if inside_points else -1.0,
                    "point_on_target_fraction":
                        float(np.mean([
                            bool(truth[tuple(np.clip(np.round(q).astype(int), 0, np.asarray(CROP)-1))])
                            for _, q in inside_points
                        ])) if inside_points else 0.0,
                })

                for p_idx, p in enumerate(cp, 1):
                    local = np.array([p["z"] - starts[0], p["y"] - starts[1], p["x"] - starts[2]], dtype=float)
                    ok = inside(local)
                    if ok:
                        q = np.clip(np.round(local).astype(int), 0, np.asarray(CROP)-1)
                        on_target = bool(truth[tuple(q)])
                        dt = nearest(local, truth)
                        dp = nearest(local, pred)
                        dtop = nearest(local, top1)
                    else:
                        on_target = False
                        dt = dp = dtop = float("inf")

                    ann_records.append({
                        "case_no": case_no,
                        "annotation_index": p_idx,
                        "class_id": c,
                        "class_name": CLASSES[c],
                        "condition": p["condition"],
                        "level": p["level"],
                        "instance_number": p["instance_number"],
                        "native_x": p["native_x"],
                        "native_y": p["native_y"],
                        "native_z": p["native_z"],
                        "full_z": p["z"],
                        "full_y": p["y"],
                        "full_x": p["x"],
                        "crop_z": local[0],
                        "crop_y": local[1],
                        "crop_x": local[2],
                        "inside_crop": int(ok),
                        "point_on_target": int(on_target),
                        "distance_to_target": dt,
                        "distance_to_threshold_prediction": dp,
                        "distance_to_top1_prediction": dtop,
                    })

    df = pd.DataFrame(records)
    adf = pd.DataFrame(ann_records)

    summary = []
    for c in FOCUS:
        g = df[df.class_id == c]
        a = adf[adf.class_id == c]
        ai = a[a.inside_crop == 1]
        summary.append({
            "class_id": c,
            "class_name": CLASSES[c],
            "mean_target_voxels": g.target_voxels.mean(),
            "mean_predicted_voxels": g.predicted_voxels.mean(),
            "mean_prediction_target_ratio": g.prediction_target_ratio.mean(),
            "mean_dice": g.dice.mean(),
            "mean_precision": g.precision.mean(),
            "mean_recall": g.recall.mean(),
            "mean_prediction_to_target_distance": g.prediction_to_target_distance.mean(),
            "mean_target_to_prediction_distance": g.target_to_prediction_distance.mean(),
            "mean_top1_to_target_distance": g.top1_to_target_distance.mean(),
            "mean_target_to_top1_distance": g.target_to_top1_distance.mean(),
            "annotation_count": len(a),
            "annotation_inside_crop": len(ai),
            "annotation_inside_fraction": div(len(ai), len(a)),
            "point_on_target_fraction": ai.point_on_target.mean() if len(ai) else 0.0,
            "mean_annotation_to_target_distance": ai.distance_to_target.replace([np.inf, -np.inf], np.nan).mean() if len(ai) else -1.0,
            "mean_annotation_to_prediction_distance": ai.distance_to_threshold_prediction.replace([np.inf, -np.inf], np.nan).mean() if len(ai) else -1.0,
            "mean_annotation_to_top1_distance": ai.distance_to_top1_prediction.replace([np.inf, -np.inf], np.nan).mean() if len(ai) else -1.0,
        })
    sdf = pd.DataFrame(summary)

    mean_ratio = sdf.mean_prediction_target_ratio.mean()
    mean_pt = sdf.mean_prediction_to_target_distance.mean()
    mean_tp = sdf.mean_target_to_prediction_distance.mean()
    mean_top = sdf.mean_top1_to_target_distance.mean()
    mean_ann_t = sdf.mean_annotation_to_target_distance.mean()
    mean_ann_p = sdf.mean_annotation_to_prediction_distance.mean()

    if mean_ratio >= 10 and mean_top >= 5:
        diagnosis = "C2_C3_SUPERVISION_OR_LOCALIZATION_QUALITY_IS_THE_DOMINANT_REMAINING_BOTTLENECK"
    elif mean_pt >= 5:
        diagnosis = "C2_C3_HIGH_PROBABILITY_ACTIVATION_IS_SPATIALLY_MISALIGNED"
    elif mean_ratio >= 10:
        diagnosis = "C2_C3_PREDICTIONS_ARE_STRONGLY_OVERSIZED_RELATIVE_TO_SUPERVISION"
    else:
        diagnosis = "C2_C3_SHOW_PARTIAL_ANATOMICAL_LOCALIZATION_BUT_REMAIN_INACCURATE"

    print("\n" + "=" * 88)
    print("PART 75.1 ANNOTATION EXTRACTION CHECK")
    print("=" * 88)
    print("Mapped annotation rows :", total_points)
    print("Cases with annotations :", f"{ann_cases}/{VAL_N}")
    if total_points == 0:
        raise RuntimeError("STOP: exact Part 67 / Part 68.1 mapping returned zero annotations.")

    print("✓ Exact annotation extraction is NON-ZERO.")

    print("\n" + "=" * 88)
    print("PART 75.1 C2/C3 SUMMARY")
    print("=" * 88)
    print(f"{'Class':<42}{'Target':>10}{'Pred':>12}{'Ratio':>10}{'Dice':>10}{'P→T':>10}{'T→P':>10}")
    for _, r in sdf.iterrows():
        print(f"C{int(r.class_id)} {r.class_name:<36}{r.mean_target_voxels:>10.1f}{r.mean_predicted_voxels:>12.1f}{r.mean_prediction_target_ratio:>10.3f}{r.mean_dice:>10.6f}{r.mean_prediction_to_target_distance:>10.3f}{r.mean_target_to_prediction_distance:>10.3f}")

    print("\n" + "=" * 88)
    print("PART 75.1 RSNA POINT / TARGET / PREDICTION AUDIT")
    print("=" * 88)
    print(f"{'Class':<42}{'Ann':>8}{'Inside':>10}{'Inside%':>10}{'Point→T':>12}{'Point→Pred':>14}{'Point→Top1':>14}")
    for _, r in sdf.iterrows():
        print(f"C{int(r.class_id)} {r.class_name:<36}{int(r.annotation_count):>8}{int(r.annotation_inside_crop):>10}{r.annotation_inside_fraction:>10.4f}{r.mean_annotation_to_target_distance:>12.3f}{r.mean_annotation_to_prediction_distance:>14.3f}{r.mean_annotation_to_top1_distance:>14.3f}")

    print("\n" + "=" * 88)
    print("PART 75.1 FINAL DIAGNOSIS")
    print("=" * 88)
    print(f"C2/C3 mean prediction/target ratio       : {mean_ratio:.3f}")
    print(f"C2/C3 mean prediction→target distance    : {mean_pt:.3f}")
    print(f"C2/C3 mean target→prediction distance    : {mean_tp:.3f}")
    print(f"C2/C3 mean top1→target distance          : {mean_top:.3f}")
    print(f"C2/C3 mean annotation→target distance    : {mean_ann_t:.3f}")
    print(f"C2/C3 mean annotation→prediction distance: {mean_ann_p:.3f}")
    print(f"Usable annotation cases                  : {ann_cases}/{VAL_N}")
    print(f"Usable annotation points                 : {total_points}")
    print(f"Diagnosis                                : {diagnosis}")

    df.to_csv(REPORT / "part75_1_case_c2_c3_localization_metrics.csv", index=False)
    sdf.to_csv(REPORT / "part75_1_c2_c3_localization_summary.csv", index=False)
    adf.to_csv(REPORT / "part75_1_annotation_level_audit.csv", index=False)

    payload = {
        "part": "75.1",
        "status": "CORRECTED",
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop_shape": CROP,
        "checkpoint": str(CKPT),
        "annotation_source": str(COORD_CSV),
        "annotation_mapping": "Exact Part 67 / Part 68.1 native -> FULL -> foreground-centered crop mapping",
        "annotation_cases": ann_cases,
        "annotation_points_total": total_points,
        "mean_c2_c3_prediction_target_ratio": float(mean_ratio),
        "mean_c2_c3_prediction_to_target_distance": float(mean_pt),
        "mean_c2_c3_target_to_prediction_distance": float(mean_tp),
        "mean_c2_c3_top1_to_target_distance": float(mean_top),
        "mean_c2_c3_annotation_to_target_distance": float(mean_ann_t),
        "mean_c2_c3_annotation_to_prediction_distance": float(mean_ann_p),
        "diagnosis": diagnosis,
        "training_performed": False,
        "part75_zero_annotation_result_reused": False,
        "scientific_limitation": "RSNA coordinates are point annotations, not manual segmentation masks.",
    }
    (REPORT / "part75_1_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "PART 75.1 — CORRECTED C2/C3 SUPERVISION + LOCALIZATION AUDIT",
        "",
        "Annotation extraction: EXACT Part 67 / Part 68.1 logic.",
        f"Validation cases: {VAL_N}",
        f"Mapped annotation points: {total_points}",
        f"Cases with annotations: {ann_cases}/{VAL_N}",
        f"C2/C3 mean prediction-target ratio: {mean_ratio:.6f}",
        f"C2/C3 mean prediction-to-target distance: {mean_pt:.6f}",
        f"C2/C3 mean target-to-prediction distance: {mean_tp:.6f}",
        f"C2/C3 mean top1-to-target distance: {mean_top:.6f}",
        f"C2/C3 mean annotation-to-target distance: {mean_ann_t:.6f}",
        f"C2/C3 mean annotation-to-prediction distance: {mean_ann_p:.6f}",
        f"Diagnosis: {diagnosis}",
        "",
        "Part 75's zero-annotation annotation audit is discarded.",
        "RSNA coordinates are point annotations, not manual segmentation masks.",
        "Point agreement is a localization/target-consistency diagnostic, not clinical segmentation accuracy.",
        "",
        sdf.to_string(index=False),
    ]
    (REPORT / "part75_1_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PART 75.1 COMPLETE")
    print("=" * 88)
    for f in sorted(REPORT.iterdir()):
        print(f)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nPART 75.1 FAILED")
        print(type(exc).__name__, str(exc))
        raise
