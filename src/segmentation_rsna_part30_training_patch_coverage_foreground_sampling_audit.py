"""
PHASE 4 - PART 30
RSNA-ONLY REAL TRAINING PATCH COVERAGE / FOREGROUND SAMPLING AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
No optimizer step is performed.
SPIDER is not used.
RSNA test set is not used.

Purpose
-------
Part 29 established that real Part 15 training targets contain foreground,
the checkpoint produces background-only predictions, and gradients reach
the model. Part 30 therefore audits the spatial/patch exposure problem.

This audit uses the exact Part 15 training cohort and the validated
Part 9 -> Part 11 loading path.

It measures:
  - full pseudo-mask foreground occupancy
  - per-class foreground occupancy
  - foreground bounding box
  - foreground depth/slice coverage
  - whether the fixed 64x96x96 model input contains foreground
  - foreground location relative to the loaded/cropped volume
  - target labels after the REAL Part 11 preprocessing
  - number of foreground-containing cases
  - approximate effective foreground exposure

IMPORTANT:
Part 30 does NOT invent a new training sampler and does NOT retrain.
It audits the tensors actually returned by the validated pipeline.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_TRAIN = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_train_cohort.csv"
)

PART15_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "checkpoints"
    / "best_model.pth"
)

PART8_TRAIN = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
    / "manifests"
    / "rsna_part8_train_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part30_training_patch_coverage_foreground_sampling_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
AUDIT_CASES = 100

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


def finite_float(x: Any) -> float:
    try:
        value = float(x)
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def validate_paths() -> None:
    banner("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 train cohort": PART15_TRAIN,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 8 train manifest": PART8_TRAIN,
    }

    missing = []

    for name, path in paths.items():
        found = path.exists()
        print(f"{name:<36}: {'FOUND' if found else 'MISSING'}")
        if not found:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required Part 30 input(s):\n" +
            "\n".join(missing)
        )


def import_part11():
    banner("IMPORTING VALIDATED PART 11")

    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "part11_corrected_part30",
        PART11_SOURCE,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not create Part 11 import specification.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    print("✓ Corrected Part 11 imported.")

    required = (
        "load_tensor_case",
        "preprocess_case",
    )

    for name in required:
        if not hasattr(module, name):
            raise AttributeError(
                f"Part 11 missing required API: {name}"
            )
        print(f"{name:<22}: {inspect.signature(getattr(module, name))}")

    return module


def import_part9():
    banner("LOADING PART 9")

    path = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"

    if not path.exists():
        raise FileNotFoundError(f"Part 9 source not found: {path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part9_part30",
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create Part 9 import specification: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    print("✓ Part 9 loader imported.")
    print(f"Part 9 source : {path}")

    return module


def load_cohort() -> pd.DataFrame:
    banner("LOADING EXACT PART 15 TRAINING COHORT")

    df = pd.read_csv(PART15_TRAIN)

    print(f"Part 15 training cohort rows : {len(df)}")
    print(f"Columns : {list(df.columns)}")

    if len(df) == 0:
        raise ValueError("Part 15 training cohort is empty.")

    return df


def tensor_to_label_volume(mask: Any) -> torch.Tensor:
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    mask = mask.detach().long()

    # Convert all expected forms to (D,H,W).
    if mask.ndim == 5:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(
                f"Unexpected 5-D target shape: {tuple(mask.shape)}"
            )
        mask = mask[0, 0]

    elif mask.ndim == 4:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            raise ValueError(
                f"Ambiguous 4-D target shape: {tuple(mask.shape)}"
            )

    elif mask.ndim != 3:
        raise ValueError(
            f"Unsupported target shape: {tuple(mask.shape)}"
        )

    return mask.cpu()


def analyze_mask(mask: torch.Tensor) -> Dict[str, Any]:
    mask = tensor_to_label_volume(mask)

    d, h, w = [int(x) for x in mask.shape]
    total = int(mask.numel())

    result: Dict[str, Any] = {
        "shape": [d, h, w],
        "total_voxels": total,
        "unique_labels": [
            int(x) for x in torch.unique(mask).tolist()
        ],
    }

    foreground = mask > 0
    fg_count = int(foreground.sum().item())

    result["foreground_voxels"] = fg_count
    result["foreground_fraction"] = (
        fg_count / total if total else 0.0
    )

    per_class = {}

    for class_id in range(NUM_CLASSES):
        voxels = int((mask == class_id).sum().item())
        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "voxels": voxels,
            "fraction": voxels / total if total else 0.0,
        }

    result["per_class"] = per_class

    if fg_count == 0:
        result["foreground_bbox"] = None
        result["foreground_depth_min"] = None
        result["foreground_depth_max"] = None
        result["foreground_depth_span"] = 0
        result["foreground_depth_fraction"] = 0.0
        result["foreground_centroid"] = None
        result["foreground_slice_counts_nonzero"] = 0
        result["foreground_slice_peak"] = 0
        return result

    coords = torch.nonzero(foreground, as_tuple=False)

    z_min, y_min, x_min = [
        int(v) for v in coords.min(dim=0).values.tolist()
    ]
    z_max, y_max, x_max = [
        int(v) for v in coords.max(dim=0).values.tolist()
    ]

    result["foreground_bbox"] = {
        "z_min": z_min,
        "z_max": z_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "depth_size": z_max - z_min + 1,
        "height_size": y_max - y_min + 1,
        "width_size": x_max - x_min + 1,
    }

    result["foreground_depth_min"] = z_min
    result["foreground_depth_max"] = z_max
    result["foreground_depth_span"] = z_max - z_min + 1
    result["foreground_depth_fraction"] = (
        (z_max - z_min + 1) / d
    )

    centroid = coords.float().mean(dim=0)

    result["foreground_centroid"] = [
        finite_float(x) for x in centroid.tolist()
    ]

    slice_counts = torch.bincount(
        coords[:, 0],
        minlength=d,
    )

    nonzero_slices = int((slice_counts > 0).sum().item())
    peak = int(slice_counts.max().item())

    result["foreground_slice_counts_nonzero"] = nonzero_slices
    result["foreground_slice_peak"] = peak

    return result


def mask_patch_coverage(
    mask: torch.Tensor,
    patch_size=PATCH_SIZE,
) -> Dict[str, Any]:
    """
    Treat the REAL loaded/preprocessed volume itself as the training
    input. Since Part 11 has already returned exactly the model patch
    shape in the existing pipeline, this audit reports whether the
    returned training tensor contains foreground.

    Additionally, when the volume is larger than the model patch,
    calculate deterministic non-overlapping patch occupancy to expose
    potential spatial sparsity. This is diagnostic only.
    """
    mask = tensor_to_label_volume(mask)

    d, h, w = [int(x) for x in mask.shape]
    pd_, ph_, pw_ = patch_size

    result = {
        "volume_shape": [d, h, w],
        "patch_size": list(patch_size),
        "returned_tensor_contains_foreground": bool(
            (mask > 0).any().item()
        ),
    }

    # Exact model tensor case.
    if (d, h, w) == tuple(patch_size):
        result.update({
            "mode": "exact_model_patch",
            "patch_count": 1,
            "foreground_patch_count": int((mask > 0).any().item()),
            "foreground_patch_fraction": float(
                (mask > 0).any().item()
            ),
            "class_patch_counts": {
                str(c): int((mask == c).any().item())
                for c in range(1, NUM_CLASSES)
            },
        })
        return result

    # Diagnostic deterministic tiling for larger volumes.
    patches_total = 0
    patches_fg = 0
    class_counts = {str(c): 0 for c in range(1, NUM_CLASSES)}

    for z0 in range(0, max(1, d - pd_ + 1), pd_):
        z1 = min(z0 + pd_, d)

        for y0 in range(0, max(1, h - ph_ + 1), ph_):
            y1 = min(y0 + ph_, h)

            for x0 in range(0, max(1, w - pw_ + 1), pw_):
                x1 = min(x0 + pw_, w)

                patch = mask[z0:z1, y0:y1, x0:x1]

                # Ignore incomplete edge patches for a strict model-size
                # interpretation.
                if tuple(patch.shape) != tuple(patch_size):
                    continue

                patches_total += 1

                if bool((patch > 0).any().item()):
                    patches_fg += 1

                for c in range(1, NUM_CLASSES):
                    if bool((patch == c).any().item()):
                        class_counts[str(c)] += 1

    result.update({
        "mode": "deterministic_nonoverlapping_tiling",
        "patch_count": patches_total,
        "foreground_patch_count": patches_fg,
        "foreground_patch_fraction": (
            patches_fg / patches_total
            if patches_total else 0.0
        ),
        "class_patch_counts": class_counts,
    })

    return result


def compare_manifest_columns(
    row: pd.Series,
    mask_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare recorded Part 15 cohort metadata with the actual loaded target.
    This is diagnostic and does not overwrite anything.
    """
    recorded_fg = row.get("foreground_voxels", None)

    try:
        recorded_fg = int(recorded_fg)
    except Exception:
        recorded_fg = None

    actual_fg = int(mask_stats["foreground_voxels"])

    return {
        "recorded_foreground_voxels": recorded_fg,
        "actual_loaded_foreground_voxels": actual_fg,
        "difference": (
            actual_fg - recorded_fg
            if recorded_fg is not None else None
        ),
        "recorded_mask_labels": str(row.get("mask_labels", "")),
        "actual_unique_labels": str(mask_stats["unique_labels"]),
        "recorded_mask_shape": str(row.get("mask_shape", "")),
        "actual_mask_shape": str(mask_stats["shape"]),
    }


def audit_case(part11, part9, row) -> Dict[str, Any]:
    image, mask, info = part11.load_tensor_case(
        row,
        part9,
    )

    if not torch.is_tensor(image):
        image = torch.as_tensor(image)

    image = image.detach().cpu().float()

    mask = tensor_to_label_volume(mask)

    mask_stats = analyze_mask(mask)
    patch_stats = mask_patch_coverage(mask)

    image_shape = list(image.shape)

    return {
        "study_id": str(row.get("study_id", "")),
        "series_id": str(row.get("series_id", "")),
        "image_shape": image_shape,
        "image_finite": bool(torch.isfinite(image).all()),
        "image_min": finite_float(image.min().item()),
        "image_max": finite_float(image.max().item()),
        "image_mean": finite_float(image.mean().item()),
        "image_std": finite_float(image.std().item()),
        "mask": mask_stats,
        "patch_coverage": patch_stats,
        "manifest_comparison": compare_manifest_columns(
            row,
            mask_stats,
        ),
        "loader_info": {
            str(k): str(v)
            for k, v in (info or {}).items()
        },
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [
        x for x in results
        if "error" not in x
    ]

    if not successful:
        return {
            "cases_attempted": len(results),
            "cases_successful": 0,
        }

    fg = [
        x["mask"]["foreground_voxels"]
        for x in successful
    ]

    fg_fraction = [
        x["mask"]["foreground_fraction"]
        for x in successful
    ]

    fg_patch = [
        x["patch_coverage"]["foreground_patch_count"]
        for x in successful
    ]

    patch_total = [
        x["patch_coverage"]["patch_count"]
        for x in successful
    ]

    loaded_fg_cases = [
        x["patch_coverage"]["returned_tensor_contains_foreground"]
        for x in successful
    ]

    class_voxels = {}
    class_cases = {}

    for c in range(1, NUM_CLASSES):
        values = [
            x["mask"]["per_class"][str(c)]["voxels"]
            for x in successful
        ]
        cases = [
            x["mask"]["per_class"][str(c)]["voxels"] > 0
            for x in successful
        ]

        class_voxels[str(c)] = {
            "class_name": CLASS_NAMES[c],
            "mean_voxels": float(np.mean(values)),
            "max_voxels": int(np.max(values)),
            "cases_with_class": int(sum(cases)),
        }

    manifest_diffs = [
        x["manifest_comparison"]["difference"]
        for x in successful
        if x["manifest_comparison"]["difference"] is not None
    ]

    return {
        "cases_attempted": len(results),
        "cases_successful": len(successful),
        "mean_foreground_voxels": float(np.mean(fg)),
        "median_foreground_voxels": float(np.median(fg)),
        "min_foreground_voxels": int(np.min(fg)),
        "max_foreground_voxels": int(np.max(fg)),
        "mean_foreground_fraction": float(np.mean(fg_fraction)),
        "cases_with_loaded_foreground": int(sum(loaded_fg_cases)),
        "cases_with_empty_loaded_target": int(
            len(successful) - sum(loaded_fg_cases)
        ),
        "mean_foreground_patches": float(np.mean(fg_patch)),
        "total_patches_audited": int(np.sum(patch_total)),
        "total_foreground_containing_patches": int(np.sum(fg_patch)),
        "foreground_patch_exposure_fraction": (
            float(np.sum(fg_patch) / np.sum(patch_total))
            if np.sum(patch_total) > 0
            else None
        ),
        "per_class": class_voxels,
        "manifest_foreground_difference_mean": (
            float(np.mean(manifest_diffs))
            if manifest_diffs else None
        ),
        "all_images_finite": all(
            x["image_finite"] for x in successful
        ),
    }


def save_results(results, aggregate_result):
    banner("SAVING PART 30 RESULTS")

    rows = []

    for r in results:
        if "error" in r:
            rows.append({
                "study_id": r.get("study_id", ""),
                "series_id": r.get("series_id", ""),
                "error": r["error"],
            })
            continue

        mask = r["mask"]
        coverage = r["patch_coverage"]
        comparison = r["manifest_comparison"]

        row = {
            "study_id": r["study_id"],
            "series_id": r["series_id"],
            "image_shape": str(r["image_shape"]),
            "image_min": r["image_min"],
            "image_max": r["image_max"],
            "image_mean": r["image_mean"],
            "image_std": r["image_std"],
            "image_finite": r["image_finite"],
            "mask_shape": str(mask["shape"]),
            "foreground_voxels": mask["foreground_voxels"],
            "foreground_fraction": mask["foreground_fraction"],
            "unique_labels": str(mask["unique_labels"]),
            "foreground_depth_min": mask["foreground_depth_min"],
            "foreground_depth_max": mask["foreground_depth_max"],
            "foreground_depth_span": mask["foreground_depth_span"],
            "foreground_depth_fraction": mask["foreground_depth_fraction"],
            "foreground_centroid": str(mask["foreground_centroid"]),
            "foreground_slice_count": mask[
                "foreground_slice_counts_nonzero"
            ],
            "foreground_slice_peak": mask["foreground_slice_peak"],
            "patch_mode": coverage["mode"],
            "patch_count": coverage["patch_count"],
            "foreground_patch_count": coverage[
                "foreground_patch_count"
            ],
            "foreground_patch_fraction": coverage[
                "foreground_patch_fraction"
            ],
            "recorded_foreground_voxels": comparison[
                "recorded_foreground_voxels"
            ],
            "actual_loaded_foreground_voxels": comparison[
                "actual_loaded_foreground_voxels"
            ],
            "manifest_foreground_difference": comparison[
                "difference"
            ],
        }

        for c in range(1, NUM_CLASSES):
            row[f"class_{c}_voxels"] = mask[
                "per_class"
            ][str(c)]["voxels"]

        rows.append(row)

    case_csv = (
        OUTPUT_DIR
        / "part30_training_patch_coverage_case_metrics.csv"
    )

    pd.DataFrame(rows).to_csv(
        case_csv,
        index=False,
    )

    summary_json = (
        OUTPUT_DIR
        / "phase4_part30_training_patch_coverage_foreground_sampling_audit_summary.json"
    )

    payload = {
        "phase": "PHASE 4 - PART 30",
        "evaluation_only": True,
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "model_weights_modified": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "aggregate": aggregate_result,
        "cases": results,
    }

    summary_json.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    report_path = (
        REPORT_DIR
        / "phase4_part30_training_patch_coverage_foreground_sampling_audit_report.txt"
    )

    report_path.write_text(
        "\n".join([
            "PHASE 4 - PART 30",
            "RSNA-ONLY REAL TRAINING PATCH COVERAGE / FOREGROUND SAMPLING AUDIT",
            "",
            "Evaluation only.",
            "No training performed.",
            "No optimizer created.",
            "No optimizer step performed.",
            "No model weights modified.",
            "SPIDER not used.",
            "RSNA test set not used.",
            "",
            "AGGREGATE RESULT",
            json.dumps(
                aggregate_result,
                indent=2,
                default=str,
            ),
            "",
            "CASE RESULTS",
            json.dumps(
                results,
                indent=2,
                default=str,
            ),
        ]),
        encoding="utf-8",
    )

    print(f"Saved: {case_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")


def main():
    banner("PHASE 4 - PART 30")
    print("RSNA-ONLY REAL TRAINING PATCH COVERAGE / FOREGROUND SAMPLING AUDIT")
    print("")
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print("")
    print("Purpose:")
    print("Determine whether the REAL Part 15 training tensors expose")
    print("foreground often enough for effective learning.")

    print(f"\nPROJECT ROOT                         : {PROJECT_ROOT}")
    print(f"RSNA DATASET                         : {RSNA_ROOT}")
    print(f"PART 11 SOURCE                       : {PART11_SOURCE}")
    print(f"PART 15 TRAIN COHORT                 : {PART15_TRAIN}")
    print(f"OUTPUT DIRECTORY                     : {OUTPUT_DIR}")

    validate_paths()

    banner("PYTORCH ENVIRONMENT")
    print(f"PyTorch version                      : {torch.__version__}")
    print(f"CUDA available                       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device                               : cuda:0")
        print(
            f"GPU                                  : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"GPU memory                            : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        print("Device                               : cpu")

    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")
    print(f"Audit cases                           : {AUDIT_CASES}")

    part11 = import_part11()
    part9 = import_part9()
    train_df = load_cohort()

    audit_df = train_df.head(
        min(AUDIT_CASES, len(train_df))
    ).copy()

    banner("STARTING REAL TRAINING PATCH COVERAGE AUDIT")
    print(f"Cases selected                       : {len(audit_df)}")

    results = []

    for index, (_, row) in enumerate(
        audit_df.iterrows(),
        start=1,
    ):
        print(
            f"\n[{index:03d}/{len(audit_df):03d}] "
            f"study={row.get('study_id', '')} "
            f"series={row.get('series_id', '')}"
        )

        try:
            result = audit_case(
                part11,
                part9,
                row,
            )

            results.append(result)

            mask = result["mask"]
            coverage = result["patch_coverage"]

            print(
                f"  image={result['image_shape']} "
                f"mask={mask['shape']}"
            )
            print(
                f"  fg_voxels={mask['foreground_voxels']} "
                f"fg_fraction={mask['foreground_fraction']:.8f} "
                f"labels={mask['unique_labels']}"
            )
            print(
                f"  bbox={mask['foreground_bbox']}"
            )
            print(
                f"  patch_mode={coverage['mode']} "
                f"patches={coverage['patch_count']} "
                f"fg_patches={coverage['foreground_patch_count']} "
                f"fg_patch_fraction={coverage['foreground_patch_fraction']:.8f}"
            )

        except Exception as exc:
            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

            results.append({
                "study_id": str(row.get("study_id", "")),
                "series_id": str(row.get("series_id", "")),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    aggregate_result = aggregate(results)

    banner("PART 30 PATCH COVERAGE SUMMARY")

    print(
        f"Cases attempted                       : "
        f"{aggregate_result['cases_attempted']}"
    )
    print(
        f"Cases successful                      : "
        f"{aggregate_result['cases_successful']}"
    )

    if aggregate_result["cases_successful"]:
        print(
            f"Mean foreground voxels               : "
            f"{aggregate_result['mean_foreground_voxels']:.4f}"
        )
        print(
            f"Median foreground voxels             : "
            f"{aggregate_result['median_foreground_voxels']:.4f}"
        )
        print(
            f"Foreground voxel fraction             : "
            f"{aggregate_result['mean_foreground_fraction']:.8f}"
        )
        print(
            f"Cases with loaded foreground          : "
            f"{aggregate_result['cases_with_loaded_foreground']}"
        )
        print(
            f"Cases with empty loaded target        : "
            f"{aggregate_result['cases_with_empty_loaded_target']}"
        )
        print(
            f"Total patches audited                 : "
            f"{aggregate_result['total_patches_audited']}"
        )
        print(
            f"Foreground-containing patches         : "
            f"{aggregate_result['total_foreground_containing_patches']}"
        )
        print(
            f"Foreground patch exposure fraction    : "
            f"{aggregate_result['foreground_patch_exposure_fraction']}"
        )

        print("\nPER-CLASS EXPOSURE")
        for c in range(1, NUM_CLASSES):
            x = aggregate_result["per_class"][str(c)]
            print(
                f"{c}: {x['class_name']:<34} "
                f"mean_voxels={x['mean_voxels']:.2f} "
                f"cases={x['cases_with_class']}"
            )

    save_results(
        results,
        aggregate_result,
    )

    banner("FINAL DECISION")

    successful = [
        x for x in results
        if "error" not in x
    ]

    if not successful:
        print(
            "INCONCLUSIVE - no real training sample could be audited."
        )

    elif aggregate_result["cases_with_empty_loaded_target"] > 0:
        print(
            "CRITICAL DATA-FLOW FINDING:"
        )
        print(
            "At least one real Part 15 training sample reaches the"
        )
        print(
            "training pipeline with an empty foreground target."
        )
        print(
            "Investigate target cropping / patch extraction before retraining."
        )

    elif (
        aggregate_result["cases_with_loaded_foreground"]
        == aggregate_result["cases_successful"]
    ):
        exposure = aggregate_result[
            "foreground_patch_exposure_fraction"
        ]

        if exposure is not None and exposure < 0.05:
            print(
                "HIGH-RISK PATCH COVERAGE FINDING:"
            )
            print(
                "Foreground is present, but estimated foreground-containing"
            )
            print(
                "patch exposure is below 5%."
            )
            print(
                "Training may be dominated by background-only patches."
            )
        else:
            print(
                "PATCH COVERAGE PASS:"
            )
            print(
                "All audited real training tensors contain foreground."
            )
            print(
                "Patch coverage alone does not explain the collapse."
            )

    else:
        print(
            "CAUTION:"
        )
        print(
            "Foreground exposure is inconsistent across audited samples."
        )

    print("")
    print("SPIDER used          : NO")
    print("Test set used        : NO")
    print("Training performed   : NO")
    print("Optimizer created    : NO")
    print("Optimizer step       : NO")
    print("Model weights changed: NO")

    banner("PHASE 4 - PART 30 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        banner("PART 30 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
