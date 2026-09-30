"""
PART 111 — RSNA VALIDATION-COHORT FULL-VOLUME INFERENCE

Purpose
-------
Run the confirmed Part 104 SwinUNETR checkpoint on the established RSNA
validation cohort using the existing project Part 9/Part 11 data-loading
pipeline.

This part:
1. Uses the Part 98 validation cohort (100 rows).
2. Imports the existing Part 9 loader and Part 11 tensor-loading path.
3. Constructs the exact MONAI SwinUNETR architecture confirmed in Parts 109-110.
4. Loads the Part 104 final checkpoint with strict=True.
5. Runs full-volume inference at (64,96,96), one case at a time.
6. Saves per-case predicted class masks as compressed NPZ files.
7. Records per-class voxel counts, foreground fraction, and pseudo-mask Dice.
8. Produces cohort-level inference summaries.

IMPORTANT
---------
- NO training.
- NO checkpoint modification.
- NO changes to src/model.py.
- Existing Part 9/Part 11 preprocessing/loading is reused.
- The validation cohort is the established Part 98 cohort.
- Dice values against the loaded masks are development/pseudo-mask metrics,
  NOT expert-ground-truth clinical segmentation performance.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PART98_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
)

VAL_COHORT = PART98_DIR / "part98_validation_cohort.csv"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

PART9_PATH = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"

# Part 11 is used because its established load_tensor_case path was used by
# the later controlled spatial-sampling/evaluation stages.
PART11_CANDIDATES = [
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py",
    SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training_corrected.py",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part111_validation_full_volume_inference"
)
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUTPUT_DIR / "part111_validation_case_inference.csv"
SUMMARY_JSON = REPORT_DIR / "part111_validation_inference_summary.json"
REPORT_TXT = REPORT_DIR / "part111_validation_inference_report.txt"


# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CLASSES = 6
FULL_SHAPE = (64, 96, 96)
FEATURE_SIZE = 12
EXPECTED_SHA256 = (
    "fa2ab2eaace097eebc488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)
# Correct canonical SHA from Part 104/109/110.
EXPECTED_SHA256 = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# HELPERS
# =============================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def banner(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def import_from_path(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_part11() -> Path:
    for path in PART11_CANDIDATES:
        if path.exists():
            return path

    # A broader search is used only if the expected historical filename is
    # absent. Prefer files whose names clearly identify Part 11 training.
    candidates = sorted(
        SRC_DIR.glob("segmentation_rsna_part11*.py"),
        key=lambda p: p.name,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No Part 11 SwinUNETR training module was found under src."
    )


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV header found: {path}")
        rows = [dict(row) for row in reader]

    if not rows:
        raise RuntimeError(f"Validation cohort is empty: {path}")

    return rows


def study_id_set(rows: List[Dict[str, str]]) -> set[str]:
    return {
        str(row.get("study_id", "")).strip()
        for row in rows
        if str(row.get("study_id", "")).strip()
    }


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint root is not a dictionary.")

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
        "weights",
    ):
        value = obj.get(key)
        if isinstance(value, dict):
            tensor_items = {
                k: v
                for k, v in value.items()
                if isinstance(k, str) and torch.is_tensor(v)
            }
            if tensor_items:
                return tensor_items

    tensor_items = {
        k: v
        for k, v in obj.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }
    if tensor_items:
        return tensor_items

    raise RuntimeError("No tensor state dictionary found.")


def build_model():
    from monai.networks.nets import SwinUNETR

    try:
        return SwinUNETR(
            in_channels=1,
            out_channels=NUM_CLASSES,
            feature_size=FEATURE_SIZE,
            spatial_dims=3,
            use_checkpoint=False,
        )
    except TypeError:
        return SwinUNETR(
            in_channels=1,
            out_channels=NUM_CLASSES,
            feature_size=FEATURE_SIZE,
            use_checkpoint=False,
        )


def normalize_case_tensor(x: Any, name: str) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.detach().cpu()

    # Established loaders may return [1,D,H,W].
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim != 3:
        raise RuntimeError(
            f"{name} must be 3D after channel squeeze; got {tuple(x.shape)}"
        )

    return x


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[int, float]:
    result: Dict[int, float] = {}

    for c in range(1, NUM_CLASSES):
        p = pred == c
        t = target == c

        p_sum = int(p.sum().item())
        t_sum = int(t.sum().item())

        if p_sum == 0 and t_sum == 0:
            result[c] = float("nan")
            continue

        intersection = int((p & t).sum().item())
        result[c] = float(
            (2.0 * intersection) / (p_sum + t_sum + 1e-8)
        )

    return result


def class_counts(pred: torch.Tensor) -> Dict[int, int]:
    return {
        c: int((pred == c).sum().item())
        for c in range(NUM_CLASSES)
    }


def safe_filename(value: str) -> str:
    value = str(value)
    chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def load_case_with_existing_pipeline(
    part11: Any,
    part9: Any,
    row: Dict[str, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reuse the established Part 11 load_tensor_case(row, part9) path.

    Historical Part 42+ scripts use this exact interface.
    """
    if not hasattr(part11, "load_tensor_case"):
        raise AttributeError(
            "Part 11 module does not expose load_tensor_case(row, part9)."
        )

    image, mask = part11.load_tensor_case(row, part9)

    image = normalize_case_tensor(image, "image").float()
    mask = normalize_case_tensor(mask, "mask").long()

    if tuple(image.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Image shape mismatch: expected {FULL_SHAPE}, got {tuple(image.shape)}"
        )

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Mask shape mismatch: expected {FULL_SHAPE}, got {tuple(mask.shape)}"
        )

    # Safety: clamp accidental out-of-range development labels rather than
    # allowing invalid labels into the per-class audit.
    if int(mask.min()) < 0 or int(mask.max()) >= NUM_CLASSES:
        raise RuntimeError(
            f"Mask labels outside 0..{NUM_CLASSES - 1}: "
            f"min={int(mask.min())}, max={int(mask.max())}"
        )

    return image, mask


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    banner("PART 111 — RSNA VALIDATION-COHORT FULL-VOLUME INFERENCE")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    overall_pass = True
    case_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Validate inputs
    # -------------------------------------------------------------------------
    banner("1. INPUT ARTIFACT VALIDATION")

    required = {
        "validation cohort": VAL_COHORT,
        "final checkpoint": CHECKPOINT,
        "Part 9 loader": PART9_PATH,
    }

    for name, path in required.items():
        print(f"{name:<25}: {'FOUND' if path.exists() else 'MISSING'}")
        if not path.exists():
            overall_pass = False

    if not overall_pass:
        print("\nFINAL STATUS: FAIL — REQUIRED_INPUT_MISSING")
        return 1

    actual_sha = sha256_file(CHECKPOINT)
    print(f"\nCheckpoint SHA256 : {actual_sha}")
    print(f"Expected SHA256   : {EXPECTED_SHA256}")

    if actual_sha != EXPECTED_SHA256:
        print("SHA256 validation : FAIL")
        print("\nFINAL STATUS: FAIL — CHECKPOINT_SHA_MISMATCH")
        return 1

    print("SHA256 validation : PASS")

    # -------------------------------------------------------------------------
    # 2. Load cohort
    # -------------------------------------------------------------------------
    banner("2. ESTABLISHED RSNA VALIDATION COHORT")

    rows = load_rows(VAL_COHORT)
    unique_studies = study_id_set(rows)

    print(f"Cohort path       : {VAL_COHORT}")
    print(f"Rows              : {len(rows)}")
    print(f"Unique study IDs  : {len(unique_studies)}")
    print(f"Duplicate rows    : {len(rows) - len(unique_studies)}")

    if len(rows) != 100:
        print(
            "WARNING: expected established Part 98 validation cohort size "
            f"of 100 rows, found {len(rows)}."
        )

    # -------------------------------------------------------------------------
    # 3. Import established loaders
    # -------------------------------------------------------------------------
    banner("3. EXISTING PROJECT DATA PIPELINE")

    part11_path = find_part11()

    print(f"Part 9 loader : {PART9_PATH}")
    print(f"Part 11 module: {part11_path}")

    try:
        part9 = import_from_path("part111_part9", PART9_PATH)
        part11 = import_from_path("part111_part11", part11_path)
        print("Part 9 import : PASS")
        print("Part 11 import: PASS")
    except Exception as exc:
        print(f"Loader import: FAIL — {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — EXISTING_DATA_PIPELINE_IMPORT_FAILED")
        return 1

    # -------------------------------------------------------------------------
    # 4. Build and load canonical model
    # -------------------------------------------------------------------------
    banner("4. CANONICAL SwinUNETR MODEL")

    try:
        model = build_model()

        params = sum(p.numel() for p in model.parameters())
        state_keys = len(model.state_dict())

        print("Architecture        : MONAI SwinUNETR")
        print("in_channels         : 1")
        print("out_channels        : 6")
        print("feature_size        : 12")
        print("spatial_dims        : 3")
        print("use_checkpoint      : False")
        print(f"Parameters          : {params:,}")
        print(f"State-dict keys     : {state_keys}")

        if params != 4_078_116 or state_keys != 159:
            raise RuntimeError(
                "Canonical model specification differs from Part 109/110."
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = extract_state_dict(checkpoint)

        if len(state_dict) != 159:
            raise RuntimeError(
                f"Checkpoint state-dict key count is {len(state_dict)}, expected 159."
            )

        model.load_state_dict(state_dict, strict=True)
        print("Strict checkpoint load: PASS")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(f"Inference device    : {device}")

    except Exception as exc:
        print(f"Model setup: FAIL — {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — MODEL_SETUP_FAILED")
        return 1

    # -------------------------------------------------------------------------
    # 5. Full validation inference
    # -------------------------------------------------------------------------
    banner("5. FULL-VOLUME VALIDATION INFERENCE")

    print(f"Target volume shape : {FULL_SHAPE}")
    print(f"Cases to process    : {len(rows)}")
    print("Batch size          : 1")
    print("Gradient tracking   : disabled")
    print("Checkpoint          : Part 104 final segmentation model")

    total_start = time.perf_counter()
    successful = 0
    failed = 0

    for idx, row in enumerate(rows):
        study_id = str(row.get("study_id", "")).strip()
        series_id = str(row.get("series_id", "")).strip()
        description = str(
            row.get("series_description", row.get("description", ""))
        ).strip()

        case_start = time.perf_counter()

        print(
            f"\n[{idx + 1:03d}/{len(rows):03d}] "
            f"study={study_id} series={series_id} "
            f"description={description}"
        )

        result: Dict[str, Any] = {
            "row_index": idx,
            "study_id": study_id,
            "series_id": series_id,
            "series_description": description,
            "status": "FAIL",
            "input_shape": None,
            "output_shape": None,
            "inference_seconds": None,
            "prediction_file": None,
            "prediction_voxels": None,
            "prediction_foreground_voxels": None,
            "prediction_foreground_fraction": None,
            "pseudo_mask_foreground_voxels": None,
            "pseudo_mask_foreground_fraction": None,
            "class_0_voxels": None,
            "class_1_voxels": None,
            "class_2_voxels": None,
            "class_3_voxels": None,
            "class_4_voxels": None,
            "class_5_voxels": None,
            "class_1_dice_pseudomask": None,
            "class_2_dice_pseudomask": None,
            "class_3_dice_pseudomask": None,
            "class_4_dice_pseudomask": None,
            "class_5_dice_pseudomask": None,
            "error": "",
        }

        try:
            image, target_mask = load_case_with_existing_pipeline(
                part11,
                part9,
                row,
            )

            result["input_shape"] = str(tuple(image.shape))

            x = image.unsqueeze(0).unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        logits = model(x)
                else:
                    logits = model(x)

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )[0].detach().cpu().long()

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            if tuple(logits.shape) != (1, NUM_CLASSES, *FULL_SHAPE):
                raise RuntimeError(
                    "Unexpected model output shape: "
                    f"{tuple(logits.shape)}"
                )

            counts = class_counts(prediction)
            dice = dice_per_class(prediction, target_mask)

            pred_fg = int((prediction > 0).sum().item())
            target_fg = int((target_mask > 0).sum().item())
            volume_voxels = int(prediction.numel())

            output_name = (
                f"{idx:03d}_study_{safe_filename(study_id)}"
                f"_series_{safe_filename(series_id)}.npz"
            )
            output_path = PRED_DIR / output_name

            np.savez_compressed(
                output_path,
                prediction=prediction.numpy().astype(np.uint8),
                study_id=np.asarray(study_id),
                series_id=np.asarray(series_id),
                series_description=np.asarray(description),
                shape=np.asarray(FULL_SHAPE, dtype=np.int32),
            )

            result.update(
                {
                    "status": "PASS",
                    "output_shape": str(tuple(logits.shape)),
                    "inference_seconds": round(float(elapsed), 4),
                    "prediction_file": str(output_path),
                    "prediction_voxels": volume_voxels,
                    "prediction_foreground_voxels": pred_fg,
                    "prediction_foreground_fraction": (
                        pred_fg / volume_voxels
                    ),
                    "pseudo_mask_foreground_voxels": target_fg,
                    "pseudo_mask_foreground_fraction": (
                        target_fg / volume_voxels
                    ),
                    "class_0_voxels": counts[0],
                    "class_1_voxels": counts[1],
                    "class_2_voxels": counts[2],
                    "class_3_voxels": counts[3],
                    "class_4_voxels": counts[4],
                    "class_5_voxels": counts[5],
                    "class_1_dice_pseudomask": dice[1],
                    "class_2_dice_pseudomask": dice[2],
                    "class_3_dice_pseudomask": dice[3],
                    "class_4_dice_pseudomask": dice[4],
                    "class_5_dice_pseudomask": dice[5],
                }
            )

            successful += 1

            print(f"  Input shape       : {tuple(image.shape)}")
            print(f"  Output shape      : {tuple(logits.shape)}")
            print(f"  Inference time    : {elapsed:.3f} sec")
            print(f"  Pred foreground   : {pred_fg:,}")
            print(f"  Pseudo foreground : {target_fg:,}")
            print(
                "  Class voxels      : "
                + ", ".join(
                    f"{c}={counts[c]:,}"
                    for c in range(NUM_CLASSES)
                )
            )
            print(
                "  Pseudo-mask Dice  : "
                + ", ".join(
                    f"C{c}={dice[c]:.4f}"
                    if not np.isnan(dice[c])
                    else f"C{c}=NaN"
                    for c in range(1, NUM_CLASSES)
                )
            )
            print(f"  Saved             : {output_path.name}")

            del x, logits, prediction

        except Exception as exc:
            failed += 1
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"  INFERENCE FAIL — {type(exc).__name__}: {exc}"
            )

        case_results.append(result)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - total_start

    # -------------------------------------------------------------------------
    # 6. Save case CSV
    # -------------------------------------------------------------------------
    banner("6. SAVE VALIDATION INFERENCE RESULTS")

    fieldnames = list(case_results[0].keys()) if case_results else []

    with CASE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_results)

    print(f"Case results CSV : {CASE_CSV}")
    print(f"Prediction dir   : {PRED_DIR}")

    # -------------------------------------------------------------------------
    # 7. Cohort summary
    # -------------------------------------------------------------------------
    banner("7. COHORT INFERENCE SUMMARY")

    passed_rows = [r for r in case_results if r["status"] == "PASS"]

    def mean_metric(key: str) -> float | None:
        values = [
            float(r[key])
            for r in passed_rows
            if r.get(key) is not None
            and not (
                isinstance(r[key], float)
                and np.isnan(r[key])
            )
        ]
        return float(np.mean(values)) if values else None

    class_dice_means = {
        f"class_{c}_dice_pseudomask_mean": mean_metric(
            f"class_{c}_dice_pseudomask"
        )
        for c in range(1, NUM_CLASSES)
    }

    class_dice_medians = {}
    for c in range(1, NUM_CLASSES):
        key = f"class_{c}_dice_pseudomask"
        vals = [
            float(r[key])
            for r in passed_rows
            if r.get(key) is not None
            and not (
                isinstance(r[key], float)
                and np.isnan(r[key])
            )
        ]
        class_dice_medians[
            f"class_{c}_dice_pseudomask_median"
        ] = float(np.median(vals)) if vals else None

    summary = {
        "part": 111,
        "status": (
            "PASS — RSNA_VALIDATION_FULL_VOLUME_INFERENCE_COMPLETED"
            if successful > 0 and failed == 0
            else (
                "PARTIAL — SOME_VALIDATION_CASES_FAILED"
                if successful > 0
                else "FAIL — NO_VALIDATION_CASES_PROCESSED"
            )
        ),
        "project_root": str(PROJECT_ROOT),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_sha,
        "validation_cohort": str(VAL_COHORT),
        "cohort_rows": len(rows),
        "unique_studies": len(unique_studies),
        "duplicate_rows": len(rows) - len(unique_studies),
        "successful_cases": successful,
        "failed_cases": failed,
        "total_inference_seconds": total_seconds,
        "mean_inference_seconds": (
            float(
                np.mean(
                    [
                        float(r["inference_seconds"])
                        for r in passed_rows
                    ]
                )
            )
            if passed_rows
            else None
        ),
        "architecture": {
            "name": "MONAI SwinUNETR",
            "in_channels": 1,
            "out_channels": 6,
            "feature_size": FEATURE_SIZE,
            "spatial_dims": 3,
            "use_checkpoint": False,
            "full_volume_shape": list(FULL_SHAPE),
            "parameters": 4_078_116,
            "state_dict_keys": 159,
        },
        "class_names": CLASS_NAMES,
        "prediction_directory": str(PRED_DIR),
        "case_results_csv": str(CASE_CSV),
        "pseudo_mask_metrics": {
            **class_dice_means,
            **class_dice_medians,
            "note": (
                "These Dice values compare predictions against the existing "
                "development/pseudo-mask targets used by the project. They "
                "are not expert-ground-truth clinical segmentation metrics."
            ),
        },
        "case_failures": [
            {
                "row_index": r["row_index"],
                "study_id": r["study_id"],
                "series_id": r["series_id"],
                "error": r["error"],
            }
            for r in case_results
            if r["status"] != "PASS"
        ],
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 8. Text report
    # -------------------------------------------------------------------------
    report_lines = [
        "PART 111 — RSNA VALIDATION-COHORT FULL-VOLUME INFERENCE",
        "=" * 72,
        "",
        f"Final status: {summary['status']}",
        "",
        "CHECKPOINT",
        f"Path: {CHECKPOINT}",
        f"SHA256: {actual_sha}",
        "",
        "ARCHITECTURE",
        "MONAI SwinUNETR",
        "in_channels=1",
        "out_channels=6",
        "feature_size=12",
        "spatial_dims=3",
        "use_checkpoint=False",
        f"full_volume={FULL_SHAPE}",
        "parameters=4,078,116",
        "state_dict_keys=159",
        "",
        "COHORT",
        f"Rows: {len(rows)}",
        f"Unique studies: {len(unique_studies)}",
        f"Duplicate rows: {len(rows) - len(unique_studies)}",
        f"Successful cases: {successful}",
        f"Failed cases: {failed}",
        "",
        "RUNTIME",
        f"Device: {device}",
        f"Total inference seconds: {total_seconds:.3f}",
        f"Mean inference seconds/case: {summary['mean_inference_seconds']}",
        "",
        "PSEUDO-MASK DICE SUMMARY",
    ]

    for c in range(1, NUM_CLASSES):
        mean_value = class_dice_means[
            f"class_{c}_dice_pseudomask_mean"
        ]
        median_value = class_dice_medians[
            f"class_{c}_dice_pseudomask_median"
        ]
        report_lines.append(
            f"Class {c} ({CLASS_NAMES[c]}): "
            f"mean={mean_value}, median={median_value}"
        )

    report_lines.extend(
        [
            "",
            "SCIENTIFIC NOTE",
            "The inference predictions are generated from the established "
            "RSNA validation cohort using the confirmed Part 104 checkpoint.",
            "The Dice values in this report are comparisons against the "
            "project's development/pseudo-mask targets and must not be "
            "presented as expert-ground-truth or clinical validation results.",
            "",
            "OUTPUTS",
            str(CASE_CSV),
            str(PRED_DIR),
            str(SUMMARY_JSON),
        ]
    )

    REPORT_TXT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 9. Final status
    # -------------------------------------------------------------------------
    banner("PART 111 FINAL RESULT")

    if successful == len(rows):
        final_status = (
            "PASS — RSNA_VALIDATION_FULL_VOLUME_INFERENCE_COMPLETED"
        )
        return_code = 0
    elif successful > 0:
        final_status = "PARTIAL — SOME_VALIDATION_CASES_FAILED"
        return_code = 1
    else:
        final_status = "FAIL — NO_VALIDATION_CASES_PROCESSED"
        return_code = 1

    print(final_status)
    print()
    print(f"Processed successfully : {successful}/{len(rows)}")
    print(f"Failed                 : {failed}")
    print(f"Total inference time   : {total_seconds:.3f} sec")
    print()
    print("OUTPUTS:")
    print(CASE_CSV)
    print(PRED_DIR)
    print(SUMMARY_JSON)
    print(REPORT_TXT)
    print("=" * 88)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
