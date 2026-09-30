"""
PART 2.26 v2 — POINT NEIGHBORHOOD / LOCAL ANATOMY AUDIT

Clean rebuild.

This version deliberately does NOT use Part 2.20B build_case_index().
The previous audit discovered that helper returns an internal list structure
that is not suitable for this audit. Validation cases are therefore built
directly from the verified Part 2.13 manifest using pandas groupby on the
real study_id + series_id columns.

No training.
No optimizer.
No checkpoint modification.
No dashboard modification.
No fabricated voxel-wise ground truth.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, sobel

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


OUTPUT_DIR = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part226_point_neighborhood_anatomy_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)
CHECKPOINT_PATH = (
    ROOT / "outputs" / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints" / "part220b_epoch_05.pth"
)

SEED = 42
VALIDATION_CASES = 25
RADII = [2, 4, 6]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

PAIR_NAMES = {
    (2, 3): "LFNN_vs_RFNN",
    (2, 4): "LFNN_vs_LSS",
    (3, 5): "RFNN_vs_RSS",
    (2, 5): "LFNN_vs_RSS",
    (3, 4): "RFNN_vs_LSS",
}


def safe_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def normalize_patch(patch):
    patch = np.asarray(patch, dtype=np.float32)
    finite = np.isfinite(patch)
    if not finite.any():
        return np.zeros_like(patch)
    vals = patch[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(patch)
    return np.clip((patch - lo) / (hi - lo), 0, 1)


def extract_patch(volume, z, y, x, radius):
    d, h, w = volume.shape
    return volume[
        max(0, z-radius):min(d, z+radius+1),
        max(0, y-radius):min(h, y+radius+1),
        max(0, x-radius):min(w, x+radius+1),
    ]


def patch_stats(patch, radius):
    p = normalize_patch(patch)
    vals = p[np.isfinite(p)]
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)

    gx = sobel(p, axis=2, mode="nearest")
    gy = sobel(p, axis=1, mode="nearest")
    gz = sobel(p, axis=0, mode="nearest")
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)

    return {
        f"r{radius}_intensity_mean": float(vals.mean()),
        f"r{radius}_intensity_std": float(vals.std()),
        f"r{radius}_intensity_median": float(np.median(vals)),
        f"r{radius}_intensity_p10": float(np.percentile(vals, 10)),
        f"r{radius}_intensity_p90": float(np.percentile(vals, 90)),
        f"r{radius}_gradient_mean": float(grad.mean()),
        f"r{radius}_gradient_std": float(grad.std()),
        f"r{radius}_gradient_p90": float(np.percentile(grad, 90)),
    }


def local_probability_stats(probs, z, y, x, radius):
    c, d, h, w = probs.shape
    z0, z1 = max(0, z-radius), min(d, z+radius+1)
    y0, y1 = max(0, y-radius), min(h, y+radius+1)
    x0, x1 = max(0, x-radius), min(w, x+radius+1)
    local = probs[:, z0:z1, y0:y1, x0:x1]
    flat = local.reshape(c, -1)
    mean_cls = flat.mean(axis=1)
    max_cls = flat.max(axis=1)
    fg = local[1:].sum(axis=0)
    return mean_cls, max_cls, float(fg.mean()), float(fg.max())


def nearest_foreground_distance(labels, z, y, x):
    fg = labels != 0
    if not fg.any():
        return float("inf")
    return float(distance_transform_edt(~fg)[z, y, x])


def nearby_counts(point_df, z, y, x, radius=6):
    point_df = point_df.copy()

    coords = point_df[
        ["model_z_float", "model_y_float", "model_x_float"]
    ].to_numpy(dtype=np.float32)

    center = np.array([z, y, x], dtype=np.float32)
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    keep = (dist <= radius) & (dist > 1e-6)
    sub = point_df.loc[keep]

    return {
        "nearby_annotation_count_r6": int(len(sub)),
        "nearby_c1_r6": int((sub["class_id"] == 1).sum()),
        "nearby_c2_r6": int((sub["class_id"] == 2).sum()),
        "nearby_c3_r6": int((sub["class_id"] == 3).sum()),
        "nearby_c4_r6": int((sub["class_id"] == 4).sum()),
        "nearby_c5_r6": int((sub["class_id"] == 5).sum()),
    }


def normalize_points(points):
    """
    Convert Part 2.20B load_case() point output into a DataFrame.
    Handles DataFrame/list/dict forms defensively without changing values.
    """
    if isinstance(points, pd.DataFrame):
        df = points.copy()
    elif isinstance(points, list):
        df = pd.DataFrame(points)
    elif isinstance(points, dict):
        df = pd.DataFrame(points)
    else:
        raise TypeError(f"Unsupported points type: {type(points)}")

    if df.empty:
        return df

    # Harmonize the model-coordinate names used by Part 2.20B.
    aliases = {
    "model_z": ["model_z", "model_z_float", "z"],
    "model_y": ["model_y", "model_y_float", "y"],
    "model_x": ["model_x", "model_x_float", "x"],
    }
    for target, candidates in aliases.items():
        if target not in df.columns:
            for candidate in candidates:
                if candidate in df.columns:
                    df[target] = df[candidate]
                    break

    # If the exact source manifest columns survived, use them.
    for col in ["class_id", "level", "condition", "series_description"]:
        if col not in df.columns:
            df[col] = ""

    required = ["class_id", "model_z", "model_y", "model_x"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Part 2.20B load_case() points missing columns: {missing}; "
            f"available={list(df.columns)}"
        )

    return df


def summarize_class(df, cid):
    sub = df[df["true_class"] == cid]
    row = {
        "class_id": cid,
        "disease": CLASS_NAMES[cid],
        "points": int(len(sub)),
        "accuracy": float(sub["correct"].mean()) if len(sub) else np.nan,
        "mean_true_probability": float(sub["true_probability"].mean()) if len(sub) else np.nan,
        "mean_local_foreground_probability": (
            float(sub["local_foreground_probability_mean"].mean())
            if len(sub) else np.nan
        ),
        "mean_nearest_foreground_distance": (
            float(sub["nearest_foreground_distance"].replace(np.inf, np.nan).mean())
            if len(sub) else np.nan
        ),
    }
    for r in RADII:
        for f in [
            "intensity_mean", "intensity_std", "intensity_median",
            "gradient_mean", "gradient_p90"
        ]:
            col = f"r{r}_{f}"
            row[col] = float(sub[col].mean()) if len(sub) else np.nan
    return row


def pair_separation(df, a, b):
    da = df[df["true_class"] == a]
    db = df[df["true_class"] == b]
    row = {
        "pair": PAIR_NAMES[(a, b)],
        "class_a": a,
        "class_a_name": CLASS_NAMES[a],
        "class_b": b,
        "class_b_name": CLASS_NAMES[b],
        "class_a_points": int(len(da)),
        "class_b_points": int(len(db)),
    }
    for r in RADII:
        for f in [
            "intensity_mean", "intensity_std", "intensity_median",
            "gradient_mean", "gradient_p90"
        ]:
            col = f"r{r}_{f}"
            ma = float(da[col].mean()) if len(da) else np.nan
            mb = float(db[col].mean()) if len(db) else np.nan
            row[f"r{r}_{f}_a_mean"] = ma
            row[f"r{r}_{f}_b_mean"] = mb
            row[f"r{r}_{f}_absolute_difference"] = (
                abs(ma - mb) if np.isfinite(ma) and np.isfinite(mb) else np.nan
            )
    return row


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 72)
    print("PART 2.26 v2 — POINT NEIGHBORHOOD / LOCAL ANATOMY AUDIT")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    # Exact Part 2.20B manifest loader and validation selector.
    manifest = part220b.load_manifest()
    print(f"Manifest rows: {len(manifest)}")
    print(
        "Annotated series: "
        f"{manifest[['study_id', 'series_id']].drop_duplicates().shape[0]}"
    )

    validation_series = part220b.select_validation_series(manifest).copy()
    validation_series["study_id"] = validation_series["study_id"].astype(str)
    validation_series["series_id"] = validation_series["series_id"].astype(str)
    print(f"Validation series selected: {len(validation_series)}")

    # Build exact series-key set, then filter the full manifest.
    valid_keys = set(
        zip(
            validation_series["study_id"],
            validation_series["series_id"],
        )
    )

    validation_manifest = manifest.copy()
    validation_manifest["study_id"] = validation_manifest["study_id"].astype(str)
    validation_manifest["series_id"] = validation_manifest["series_id"].astype(str)

    validation_manifest = validation_manifest[
        validation_manifest.apply(
            lambda r: (r["study_id"], r["series_id"]) in valid_keys,
            axis=1,
        )
    ].copy()

    # Explicitly reject accidental header rows / non-numeric IDs.
    bad_ids = validation_manifest[
        ~validation_manifest["study_id"].str.fullmatch(r"\d+")
        | ~validation_manifest["series_id"].str.fullmatch(r"\d+")
    ]
    if not bad_ids.empty:
        raise RuntimeError(
            "Validation manifest contains non-numeric study/series IDs. "
            f"Examples:\n{bad_ids[['study_id','series_id']].head().to_string(index=False)}"
        )

    print(f"Validation annotation rows: {len(validation_manifest)}")
    print(f"Validation studies: {validation_manifest['study_id'].nunique()}")

    # DIRECT GROUPING — no build_case_index().
    case_groups = list(
        validation_manifest.groupby(
            ["study_id", "series_id"],
            sort=True,
        )
    )

    if len(case_groups) != VALIDATION_CASES:
        raise RuntimeError(
            f"Expected {VALIDATION_CASES} validation cases, "
            f"constructed {len(case_groups)}."
        )

    print(f"Validation cases constructed: {len(case_groups)}")

    # Exact Part 2.20B model/checkpoint.
    model = part220b.build_model()
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint missing keys: {len(missing)}")
    print(f"Checkpoint unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError("Checkpoint/model mismatch.")

    model = model.to(device)
    model.eval()

    results = []
    failed = []

    for case_number, ((study_id, series_id), point_df) in enumerate(
        case_groups, start=1
    ):
        study_id = str(study_id)
        series_id = str(series_id)

        print(
            f"[{case_number:02d}/{len(case_groups):02d}] "
            f"study={study_id} series={series_id} points={len(point_df)}"
        )

        try:
            image, points, geometry = part220b.load_case(
                study_id,
                series_id,
                point_df,
            )
            image = np.asarray(image, dtype=np.float32)
            point_df_loaded = normalize_points(points)

            with torch.no_grad():
                tensor = (
                    torch.from_numpy(image)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            labels = np.argmax(probs, axis=0).astype(np.int16)

            for _, p in point_df_loaded.iterrows():
                true_class = int(p["class_id"])
                z = int(np.clip(round(float(p["model_z"])), 0, image.shape[0]-1))
                y = int(np.clip(round(float(p["model_y"])), 0, image.shape[1]-1))
                x = int(np.clip(round(float(p["model_x"])), 0, image.shape[2]-1))

                pred_class = int(labels[z, y, x])
                mean_probs, max_probs, fg_mean, fg_max = local_probability_stats(
                    probs, z, y, x, max(RADII)
                )

                row = {
                    "study_id": study_id,
                    "series_id": series_id,
                    "series_description": str(p.get("series_description", "")),
                    "condition": str(p.get("condition", "")),
                    "level": str(p.get("level", "")),
                    "true_class": true_class,
                    "true_name": CLASS_NAMES.get(true_class, str(true_class)),
                    "predicted_class": pred_class,
                    "predicted_name": CLASS_NAMES.get(pred_class, str(pred_class)),
                    "correct": int(pred_class == true_class),
                    "model_z": float(p["model_z"]),
                    "model_y": float(p["model_y"]),
                    "model_x": float(p["model_x"]),
                    "patient_x": safe_float(p.get("patient_x", np.nan)),
                    "patient_y": safe_float(p.get("patient_y", np.nan)),
                    "patient_z": safe_float(p.get("patient_z", np.nan)),
                    "true_probability": float(probs[true_class, z, y, x]),
                    "predicted_probability": float(probs[pred_class, z, y, x]),
                    "local_foreground_probability_mean": fg_mean,
                    "local_foreground_probability_max": fg_max,
                    "local_mean_predicted_class": int(np.argmax(mean_probs)),
                    "local_mean_predicted_class_probability": float(np.max(mean_probs)),
                    "nearest_foreground_distance": nearest_foreground_distance(
                        labels, z, y, x
                    ),
                    "voxel_z": z,
                    "voxel_y": y,
                    "voxel_x": x,
                }

                row.update(nearby_counts(
                    point_df,
                    float(p["model_z"]),
                    float(p["model_y"]),
                    float(p["model_x"]),
                ))

                for radius in RADII:
                    row.update(
                        patch_stats(
                            extract_patch(image, z, y, x, radius),
                            radius,
                        )
                    )

                results.append(row)

        except Exception as exc:
            failed.append({
                "study_id": study_id,
                "series_id": series_id,
                "error": repr(exc),
            })
            print(f"  ERROR: {exc}")

    results_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    if results_df.empty:
        raise RuntimeError(
            "No point-level results were produced. "
            "Check the failed-series output and DICOM paths."
        )

    disease_df = pd.DataFrame([
        summarize_class(results_df, cid)
        for cid in [1, 2, 3, 4, 5]
    ])

    pair_df = pd.DataFrame([
        pair_separation(results_df, a, b)
        for a, b in PAIR_NAMES
    ])

    error_pairs = (
        results_df[results_df["correct"] == 0]
        .groupby(["true_name", "predicted_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    level_df = (
        results_df.groupby("level", dropna=False)
        .agg(
            points=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_true_probability=("true_probability", "mean"),
            mean_local_foreground_probability=(
                "local_foreground_probability_mean", "mean"
            ),
        )
        .reset_index()
    )

    # Save.
    results_df.to_csv(
        OUTPUT_DIR / "part226_point_neighborhood_features.csv",
        index=False,
    )
    disease_df.to_csv(
        OUTPUT_DIR / "part226_disease_neighborhood_summary.csv",
        index=False,
    )
    pair_df.to_csv(
        OUTPUT_DIR / "part226_disease_pair_separation.csv",
        index=False,
    )
    error_pairs.to_csv(
        OUTPUT_DIR / "part226_error_pairs.csv",
        index=False,
    )
    level_df.to_csv(
        OUTPUT_DIR / "part226_level_summary.csv",
        index=False,
    )
    failed_df.to_csv(
        OUTPUT_DIR / "part226_failed_series.csv",
        index=False,
    )

    overall = float(results_df["correct"].mean())
    macro = float(disease_df["accuracy"].mean())
    mean_prob = float(results_df["true_probability"].mean())
    hit = float((results_df["true_probability"] >= 0.5).mean())

    summary = {
        "part": "2.26 v2",
        "status": "COMPLETE",
        "validation_cases": len(case_groups),
        "validation_studies": int(validation_manifest["study_id"].nunique()),
        "audited_points": len(results_df),
        "failed_series": len(failed_df),
        "overall_point_accuracy": overall,
        "macro_disease_accuracy": macro,
        "mean_true_class_probability": mean_prob,
        "hit_rate_at_0_50": hit,
        "neighborhood_radii": RADII,
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "voxel_ground_truth_fabricated": False,
        "annotation_type": "RSNA point/localization annotations",
    }

    (OUTPUT_DIR / "part226_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = [
        "PART 2.26 v2 — POINT NEIGHBORHOOD / LOCAL ANATOMY AUDIT",
        "",
        f"Validation cases: {len(case_groups)}",
        f"Validation studies: {validation_manifest['study_id'].nunique()}",
        f"Audited points: {len(results_df)}",
        f"Failed series: {len(failed_df)}",
        f"Overall point accuracy: {overall:.12f}",
        f"Macro disease accuracy: {macro:.12f}",
        f"Mean true-class probability: {mean_prob:.12f}",
        f"Hit rate @0.50: {hit:.12f}",
        "",
        "DISEASE NEIGHBORHOOD SUMMARY",
        disease_df.to_string(index=False),
        "",
        "DISEASE PAIR LOCAL FEATURE SEPARATION",
        pair_df.to_string(index=False),
        "",
        "TOP ERROR PAIRS",
        error_pairs.head(20).to_string(index=False),
        "",
        "LEVEL SUMMARY",
        level_df.to_string(index=False),
        "",
        "Training performed: NO",
        "Checkpoint modified: NO",
        "Dashboard modified: NO",
        "Voxel ground truth fabricated: NO",
        "",
        "RSNA annotations are point/localization annotations, not manual",
        "voxel-wise segmentation masks. Neighborhood features are descriptive.",
    ]

    (REPORT_DIR / "part226_report.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("PART 2.26 v2 COMPLETE")
    print("=" * 72)
    print(f"Processed cases : {len(case_groups) - len(failed_df)}/{len(case_groups)}")
    print(f"Audited points  : {len(results_df)}")
    print(f"Overall accuracy: {overall:.12f}")
    print(f"Macro accuracy  : {macro:.12f}")
    print(f"Mean probability: {mean_prob:.12f}")
    print(f"Hit rate @0.50  : {hit:.12f}")
    print(f"Outputs         : {OUTPUT_DIR}")
    print("")
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")
    print("Voxel ground truth fabricated: NO")


if __name__ == "__main__":
    main()
