"""
PART 2.24 — EXACT PIPELINE PREDICTION VISUALIZATION AUDIT

Purpose
-------
Visualize predictions produced by the ACTUAL Part 2.20B implementation.
This script intentionally imports the real Part 2.20B module and reuses:
    load_manifest()
    select_validation_series()
    build_case_index()
    load_case()
    evaluate_case()
    build_model()

No training is performed.
No checkpoint is modified.
No dashboard is modified.
No voxel-wise ground truth is fabricated.

The goal is to create visual evidence corresponding exactly to the
validated Part 2.20B Epoch-5 result reproduced by Part 2.23.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

PART220B_PATH = SRC_DIR / "segmentation_rsna_part220b_geometry_corrected_training.py"
CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part220b_geometry_corrected_training"
    / "checkpoints"
    / "part220b_epoch_05.pth"
)
MANIFEST = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
    / "part213_point_manifest.csv"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part224_exact_pipeline_visualization_audit"
)
VIS_DIR = OUTPUT_DIR / "visualizations"

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


def load_part220b():
    spec = importlib.util.spec_from_file_location("part220b_exact", PART220B_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Part 2.20B from {PART220B_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["part220b_exact"] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_for_display(volume):
    arr = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[finite]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def extract_case_prediction(part220b, model, case):
    """
    Use the actual Part 2.20B evaluate_case() exactly.

    The returned prediction dictionary is intentionally treated as the
    source of truth for point predictions.
    """
    result = part220b.evaluate_case(
        model=model,
        image=case["image"],
        points=case["points"],
        device=DEVICE,
    )
    return result


def unpack_evaluation(result):
    """
    Part 2.20B evaluate_case() returns point rows plus foreground ratio.

    Keep this tolerant of the exact container shape while never changing
    the prediction values.
    """
    if isinstance(result, tuple):
        if len(result) >= 2:
            return result[0], result[1]
    if isinstance(result, dict):
        rows = result.get("point_results")
        if rows is None:
            rows = result.get("results")
        fg = result.get("foreground_ratio", np.nan)
        return rows, fg
    return result, np.nan


def point_rows_to_dataframe(rows):
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    elif isinstance(rows, list):
        df = pd.DataFrame(rows)
    elif isinstance(rows, dict):
        df = pd.DataFrame(rows)
    else:
        raise TypeError(f"Unsupported point-result type: {type(rows)}")

    # Normalize likely column names without changing semantics.
    aliases = {
        "true": "true_class",
        "target_class": "true_class",
        "pred": "predicted_class",
        "prediction": "predicted_class",
        "probability": "true_probability",
        "true_prob": "true_probability",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    required = {"true_class", "predicted_class"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Part 2.20B evaluation result does not expose {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def choose_display_slice(image, point):
    """
    Choose the depth slice associated with a point.

    The image is the exact Part 2.20B canonical model input. This is only
    a visualization choice; it does not alter evaluation.
    """
    vol = np.asarray(image)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {vol.shape}")

    p = np.asarray(point, dtype=float).reshape(-1)
    if len(p) < 3:
        z = vol.shape[0] // 2
    else:
        z = int(np.clip(round(p[0]), 0, vol.shape[0] - 1))
    return z


def get_point_coordinate(row):
    candidates = [
        ("z", "y", "x"),
        ("model_z", "model_y", "model_x"),
        ("model_z_float", "model_y_float", "model_x_float"),
        ("canonical_z", "canonical_y", "canonical_x"),
    ]
    for names in candidates:
        if all(name in row.index for name in names):
            return np.array([safe_float(row[n]) for n in names], dtype=float)

    return None


def draw_case(case_index, study_id, series_id, case, point_df, foreground_ratio, out_path):
    image = np.asarray(case["image"], dtype=np.float32)
    display_volume = normalize_for_display(image)

    rows = []

    for _, row in point_df.iterrows():
        point = get_point_coordinate(row)
        if point is None:
            continue

        z = choose_display_slice(display_volume, point)
        y = int(np.clip(round(point[1]), 0, display_volume.shape[1] - 1))
        x = int(np.clip(round(point[2]), 0, display_volume.shape[2] - 1))

        true_class = int(row["true_class"])
        pred_class = int(row["predicted_class"])
        correct = true_class == pred_class

        true_prob = safe_float(
            row.get("true_probability", row.get("correct_class_probability", np.nan))
        )

        rows.append(
            {
                "row": row,
                "point": point,
                "z": z,
                "x": x,
                "y": y,
                "true_class": true_class,
                "pred_class": pred_class,
                "true_prob": true_prob,
                "correct": correct,
            }
        )

    if not rows:
        return False

    # Prefer a representative point: correct example first, then highest
    # true-class probability.
    correct_rows = [r for r in rows if r["correct"]]
    if correct_rows:
        selected = max(
            correct_rows,
            key=lambda r: np.nan_to_num(r["true_prob"], nan=-1.0),
        )
    else:
        selected = max(
            rows,
            key=lambda r: np.nan_to_num(r["true_prob"], nan=-1.0),
        )

    z = selected["z"]
    x = selected["x"]
    y = selected["y"]

    # Orthogonal views of the exact canonical model input.
    axial = display_volume[z, :, :]
    sagittal = display_volume[:, :, x]
    coronal = display_volume[:, y, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial, cmap="gray")
    axes[0].scatter([x], [y], s=70, marker="+")
    axes[0].set_title(f"Axial z={z}")

    axes[1].imshow(sagittal, cmap="gray", aspect="auto")
    axes[1].scatter([y], [z], s=70, marker="+")
    axes[1].set_title(f"Sagittal x={x}")

    axes[2].imshow(coronal, cmap="gray", aspect="auto")
    axes[2].scatter([x], [z], s=70, marker="+")
    axes[2].set_title(f"Coronal y={y}")

    true_name = CLASS_NAMES.get(selected["true_class"], str(selected["true_class"]))
    pred_name = CLASS_NAMES.get(selected["pred_class"], str(selected["pred_class"]))

    status = "CORRECT" if selected["correct"] else "INCORRECT"
    prob_text = (
        f"{selected['true_prob']:.4f}"
        if np.isfinite(selected["true_prob"])
        else "N/A"
    )

    fig.suptitle(
        f"Part 2.20B Exact Prediction Audit — {status}\n"
        f"Study {study_id} | Series {series_id}\n"
        f"True: {true_name} | Predicted: {pred_name} | "
        f"True-class probability: {prob_text}\n"
        f"Foreground ratio: {float(foreground_ratio):.6f}",
        fontsize=11,
    )

    for ax in axes:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PART 2.24 — EXACT PIPELINE PREDICTION VISUALIZATION AUDIT")
    print("=" * 78)
    print(f"Project root: {ROOT}")
    print(f"Part 2.20B source: {PART220B_PATH}")
    print(f"Manifest: {MANIFEST}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not PART220B_PATH.exists():
        raise FileNotFoundError(PART220B_PATH)
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    part220b = load_part220b()

    required = [
        "load_manifest",
        "select_validation_series",
        "build_case_index",
        "load_case",
        "evaluate_case",
        "build_model",
    ]
    for name in required:
        if not hasattr(part220b, name):
            raise AttributeError(f"Part 2.20B missing required function: {name}")

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Part 2.20B load_manifest() takes no arguments and uses its own
    # MANIFEST_PATH constant. This is intentional: we must reproduce the
    # exact implementation rather than silently changing its data source.
    manifest = part220b.load_manifest()
    validation_series = part220b.select_validation_series(manifest)

    print(f"Manifest rows: {len(manifest)}")
    print(f"Annotated series: {manifest[['study_id','series_id']].drop_duplicates().shape[0]}")
    print(f"Validation series selected: {len(validation_series)}")

    # Build the exact validation case objects used by Part 2.20B.
    validation_keys = {
        (str(row["study_id"]), str(row["series_id"]))
        for _, row in validation_series.iterrows()
    }
    validation_manifest = manifest[
        manifest.apply(
            lambda row: (
                str(row["study_id"]),
                str(row["series_id"]),
            ) in validation_keys,
            axis=1,
        )
    ].copy()
    validation_cases = part220b.build_case_index(validation_manifest)

    model = part220b.build_model()
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    load_result = model.load_state_dict(state_dict, strict=True)
    print("Part 2.20B Epoch-5 checkpoint loaded successfully.")
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

    model.to(DEVICE)
    model.eval()

    all_rows = []
    visualizations = []

    # Visualize every validation case. The representative point selected
    # inside draw_case is deterministic and does not alter metrics.
    for idx, case_entry in enumerate(validation_cases, start=1):
        study_id = str(case_entry["study_id"])
        series_id = str(case_entry["series_id"])

        print(f"[{idx:02d}/{len(validation_cases)}] {study_id}/{series_id}")

        # Exact Part 2.20B load_case() contract:
        #     load_case(study_id, series_id, point_df)
        image, points, geometry = part220b.load_case(
            study_id,
            series_id,
            case_entry["points"],
        )

        case = {
            "image": image,
            "points": points,
            "geometry": geometry,
        }

        # Exact Part 2.20B evaluate_case() contract:
        #     evaluate_case(model, image, points)
        evaluation = part220b.evaluate_case(
            model,
            image,
            points,
        )
        point_rows, foreground_ratio = unpack_evaluation(evaluation)
        df = point_rows_to_dataframe(point_rows)

        if "study_id" not in df.columns:
            df["study_id"] = study_id
        if "series_id" not in df.columns:
            df["series_id"] = series_id

        df["foreground_ratio"] = float(foreground_ratio)
        df["correct"] = df["true_class"].astype(int) == df["predicted_class"].astype(int)

        all_rows.append(df)

        out_path = VIS_DIR / f"{idx:02d}_{study_id}_{series_id}.png"
        try:
            made = draw_case(
                case_index=idx,
                study_id=study_id,
                series_id=series_id,
                case=case,
                point_df=df,
                foreground_ratio=foreground_ratio,
                out_path=out_path,
            )
            if made:
                visualizations.append(str(out_path))
        except Exception as exc:
            print(f"  Visualization warning: {exc}")

    if not all_rows:
        raise RuntimeError("No validation predictions were produced.")

    results = pd.concat(all_rows, ignore_index=True)

    results["true_name"] = results["true_class"].map(CLASS_NAMES)
    results["predicted_name"] = results["predicted_class"].map(CLASS_NAMES)

    point_csv = OUTPUT_DIR / "part224_exact_point_predictions.csv"
    results.to_csv(point_csv, index=False)

    disease_summary = (
        results.groupby("true_class")
        .agg(
            points=("true_class", "size"),
            accuracy=("correct", "mean"),
            mean_true_probability=(
                "true_probability",
                "mean",
            )
            if "true_probability" in results.columns
            else ("correct", "mean"),
        )
        .reset_index()
    )
    disease_summary["class_name"] = disease_summary["true_class"].map(CLASS_NAMES)

    disease_csv = OUTPUT_DIR / "part224_disease_summary.csv"
    disease_summary.to_csv(disease_csv, index=False)

    overall_accuracy = float(results["correct"].mean())

    disease_acc = (
        results.groupby("true_class")["correct"].mean().reindex([1, 2, 3, 4, 5])
    )
    macro_accuracy = float(disease_acc.mean())

    if "true_probability" in results.columns:
        mean_probability = float(results["true_probability"].mean())
        hit_rate = float((results["true_probability"] >= 0.50).mean())
    else:
        mean_probability = float("nan")
        hit_rate = float("nan")

    summary = {
        "part": "2.24",
        "status": "COMPLETE",
        "purpose": "Exact Part 2.20B prediction visualization audit",
        "validation_cases": int(len(validation_series)),
        "audited_points": int(len(results)),
        "overall_accuracy": overall_accuracy,
        "macro_disease_accuracy": macro_accuracy,
        "mean_true_class_probability": mean_probability,
        "hit_rate_at_0_50": hit_rate,
        "visualizations": len(visualizations),
        "checkpoint": str(CHECKPOINT),
        "training_performed": False,
        "checkpoint_modified": False,
        "dashboard_modified": False,
        "voxel_ground_truth_fabricated": False,
        "important_note": (
            "Predictions and case loading come directly from the actual "
            "Part 2.20B implementation. Visualization does not alter "
            "evaluation."
        ),
    }

    summary_path = OUTPUT_DIR / "part224_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = OUTPUT_DIR / "part224_report.txt"
    report.write_text(
        "\n".join(
            [
                "PART 2.24 — EXACT PIPELINE PREDICTION VISUALIZATION AUDIT",
                "",
                f"Validation cases: {len(validation_series)}",
                f"Audited points: {len(results)}",
                f"Overall accuracy: {overall_accuracy:.12f}",
                f"Macro disease accuracy: {macro_accuracy:.12f}",
                f"Mean true-class probability: {mean_probability:.12f}",
                f"Hit rate @0.50: {hit_rate:.12f}",
                f"Visualizations: {len(visualizations)}",
                "",
                "Training performed: NO",
                "Checkpoint modified: NO",
                "Dashboard modified: NO",
                "Voxel ground truth fabricated: NO",
                "",
                "The exact Part 2.20B implementation supplied the case loading",
                "and prediction values. This part only creates visual audit",
                "outputs from those exact predictions.",
            ]
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PART 2.24 COMPLETE")
    print("=" * 78)
    print(f"Processed cases : {len(validation_cases)}/{len(validation_cases)}")
    print(f"Audited points  : {len(results)}")
    print(f"Overall accuracy: {overall_accuracy:.12f}")
    print(f"Macro accuracy  : {macro_accuracy:.12f}")
    print(f"Mean probability: {mean_probability:.12f}")
    print(f"Hit rate @0.50  : {hit_rate:.12f}")
    print(f"Visualizations  : {len(visualizations)}")
    print(f"Outputs         : {OUTPUT_DIR}")
    print()
    print("Training performed: NO")
    print("Checkpoint modified: NO")
    print("Dashboard modified: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
