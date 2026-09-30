"""
PART 112 — FIXED RSNA VALIDATION FULL-VOLUME INFERENCE

Purpose
-------
Correct Part 111's data-loader invocation without changing the model,
checkpoint, cohort, or established preprocessing pipeline.

Root cause found from the historical Part 11 source:
    load_tensor_case(part9, row)
returns:
    image_tensor, mask_tensor, info

Part 111 incorrectly called:
    load_tensor_case(row, part9)

It also passed CSV rows as plain dictionaries, while the established
Part 9/Part 11 pipeline expects pandas.Series rows.

This part therefore:
- converts each validation row to pandas.Series;
- calls Part 11 exactly as historically defined: (part9, row);
- preserves the established Part 11 preprocessing;
- uses the confirmed Part 104 checkpoint;
- uses canonical MONAI SwinUNETR 1/6/12;
- performs inference only;
- saves predictions and case/cohort reports.

NO TRAINING.
NO CHECKPOINT MODIFICATION.
NO src/model.py MODIFICATION.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

VAL_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
    / "part98_validation_cohort.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

PART9_PATH = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11_PATH = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part112_validation_full_volume_inference_fixed"
)
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUTPUT_DIR / "part112_validation_case_inference.csv"
SUMMARY_JSON = REPORT_DIR / "part112_validation_inference_summary.json"
REPORT_TXT = REPORT_DIR / "part112_validation_inference_report.txt"


# =============================================================================
# CANONICAL CONFIGURATION
# =============================================================================
NUM_CLASSES = 6
FEATURE_SIZE = 12
FULL_SHAPE = (64, 96, 96)

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


# =============================================================================
# UTILITIES
# =============================================================================
def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(
        ch if (ch.isalnum() or ch in "-_.") else "_"
        for ch in text
    )


def normalize_image(image: Any) -> torch.Tensor:
    image = torch.as_tensor(image).float().detach().cpu()

    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim == 4 and image.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected image tensor shape: {tuple(image.shape)}"
        )

    if image.ndim != 4:
        raise RuntimeError(
            f"Expected Part 11 image shape [1,D,H,W], got {tuple(image.shape)}"
        )

    if tuple(image.shape[-3:]) != FULL_SHAPE:
        raise RuntimeError(
            f"Expected image volume {FULL_SHAPE}, "
            f"got {tuple(image.shape[-3:])}"
        )

    return image.contiguous()


def normalize_mask(mask: Any) -> torch.Tensor:
    mask = torch.as_tensor(mask).long().detach().cpu()

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim != 3:
        raise RuntimeError(
            f"Expected Part 11 mask shape [D,H,W], got {tuple(mask.shape)}"
        )

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Expected mask volume {FULL_SHAPE}, got {tuple(mask.shape)}"
        )

    labels = torch.unique(mask).tolist()
    if any(int(x) < 0 or int(x) >= NUM_CLASSES for x in labels):
        raise RuntimeError(
            f"Invalid mask labels: {sorted(int(x) for x in labels)}"
        )

    return mask.contiguous()


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint root is not a dictionary.")

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
        "weights",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state = {
                k: v
                for k, v in value.items()
                if isinstance(k, str) and torch.is_tensor(v)
            }
            if state:
                return state

    state = {
        k: v
        for k, v in checkpoint.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }

    if state:
        return state

    raise RuntimeError("No tensor state dictionary found.")


def dice_for_class(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int,
) -> float:
    pred = prediction == class_id
    true = target == class_id

    pred_sum = int(pred.sum())
    true_sum = int(true.sum())

    if pred_sum == 0 and true_sum == 0:
        return float("nan")

    intersection = int((pred & true).sum())

    return float(
        (2.0 * intersection)
        / (pred_sum + true_sum + 1e-8)
    )


def class_counts(prediction: torch.Tensor) -> Dict[int, int]:
    return {
        c: int((prediction == c).sum())
        for c in range(NUM_CLASSES)
    }


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    banner("PART 112 — FIXED RSNA VALIDATION FULL-VOLUME INFERENCE")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # 1. Inputs
    # -------------------------------------------------------------------------
    banner("1. INPUT VALIDATION")

    required = {
        "validation cohort": VAL_COHORT,
        "final checkpoint": CHECKPOINT,
        "Part 9 loader": PART9_PATH,
        "Part 11 loader": PART11_PATH,
    }

    for name, path in required.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<25}: {status}")
        if not path.exists():
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
    # 2. Cohort
    # -------------------------------------------------------------------------
    banner("2. VALIDATION COHORT")

    cohort_df = pd.read_csv(VAL_COHORT)

    if cohort_df.empty:
        print("\nFINAL STATUS: FAIL — EMPTY_VALIDATION_COHORT")
        return 1

    print(f"Cohort rows       : {len(cohort_df)}")
    print(
        f"Unique studies   : "
        f"{cohort_df['study_id'].nunique() if 'study_id' in cohort_df else 'unknown'}"
    )

    if len(cohort_df) != 100:
        print(
            f"WARNING: expected 100 established Part 98 rows; "
            f"found {len(cohort_df)}."
        )

    # -------------------------------------------------------------------------
    # 3. Existing modules
    # -------------------------------------------------------------------------
    banner("3. EXISTING PART 9 / PART 11 PIPELINE")

    try:
        part9 = import_from_path("part112_part9", PART9_PATH)
        part11 = import_from_path("part112_part11", PART11_PATH)
    except Exception as exc:
        print(
            f"Module import failed: {type(exc).__name__}: {exc}"
        )
        print("\nFINAL STATUS: FAIL — PIPELINE_IMPORT_FAILED")
        return 1

    print("Part 9 import  : PASS")
    print("Part 11 import : PASS")

    if not hasattr(part9, "load_case"):
        print("Part 9 load_case: MISSING")
        return 1

    if not hasattr(part11, "load_tensor_case"):
        print("Part 11 load_tensor_case: MISSING")
        return 1

    print(
        "Part 11 contract: load_tensor_case(part9, row) "
        "→ image, mask, info"
    )

    # -------------------------------------------------------------------------
    # 4. Canonical model
    # -------------------------------------------------------------------------
    banner("4. CANONICAL SwinUNETR")

    try:
        from monai.networks.nets import SwinUNETR

        try:
            model = SwinUNETR(
                in_channels=1,
                out_channels=NUM_CLASSES,
                feature_size=FEATURE_SIZE,
                spatial_dims=3,
                use_checkpoint=False,
            )
        except TypeError:
            model = SwinUNETR(
                in_channels=1,
                out_channels=NUM_CLASSES,
                feature_size=FEATURE_SIZE,
                use_checkpoint=False,
            )

        parameter_count = sum(
            p.numel() for p in model.parameters()
        )
        state_key_count = len(model.state_dict())

        print("Architecture    : MONAI SwinUNETR")
        print("in_channels     : 1")
        print("out_channels    : 6")
        print("feature_size    : 12")
        print("spatial_dims    : 3")
        print("use_checkpoint  : False")
        print(f"Parameters      : {parameter_count:,}")
        print(f"State keys      : {state_key_count}")

        if parameter_count != 4_078_116:
            raise RuntimeError(
                f"Unexpected parameter count: {parameter_count}"
            )

        if state_key_count != 159:
            raise RuntimeError(
                f"Unexpected state-key count: {state_key_count}"
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
        state = extract_state_dict(checkpoint)

        cleaned_state = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state.items()
        }

        model.load_state_dict(
            cleaned_state,
            strict=True,
        )

        print("Strict checkpoint load: PASS")

    except Exception as exc:
        print(
            f"Model setup failed: {type(exc).__name__}: {exc}"
        )
        print("\nFINAL STATUS: FAIL — MODEL_SETUP_FAILED")
        return 1

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    print(f"Inference device: {device}")

    # -------------------------------------------------------------------------
    # 5. Probe first case before running all 100
    # -------------------------------------------------------------------------
    banner("5. FIRST-CASE DATA PIPELINE PROBE")

    first_row = cohort_df.iloc[0]

    print(
        f"Study ID : {first_row.get('study_id', '')}"
    )
    print(
        f"Series ID: {first_row.get('series_id', '')}"
    )
    print(
        f"Series   : {first_row.get('series_description', '')}"
    )

    try:
        # CRITICAL FIX:
        # Part 11 signature is load_tensor_case(part9, row), not
        # load_tensor_case(row, part9).
        probe_loaded = part11.load_tensor_case(
            part9,
            first_row,
        )

        if not isinstance(probe_loaded, (tuple, list)):
            raise RuntimeError(
                "Part 11 load_tensor_case did not return tuple/list."
            )

        if len(probe_loaded) < 2:
            raise RuntimeError(
                "Part 11 load_tensor_case returned fewer than two values."
            )

        probe_image = normalize_image(probe_loaded[0])
        probe_mask = normalize_mask(probe_loaded[1])

        print(
            f"Image tensor : {tuple(probe_image.shape)}"
        )
        print(
            f"Mask tensor  : {tuple(probe_mask.shape)}"
        )
        print(
            f"Image range  : "
            f"{float(probe_image.min()):.5f} → "
            f"{float(probe_image.max()):.5f}"
        )
        print(
            f"Mask labels  : "
            f"{sorted(int(x) for x in torch.unique(probe_mask))}"
        )

        x_probe = probe_image.unsqueeze(0).to(device)

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    y_probe = model(x_probe)
            else:
                y_probe = model(x_probe)

        print(
            f"Forward output: {tuple(y_probe.shape)}"
        )

        expected_output = (
            1,
            NUM_CLASSES,
            *FULL_SHAPE,
        )

        if tuple(y_probe.shape) != expected_output:
            raise RuntimeError(
                f"Unexpected output shape: {tuple(y_probe.shape)}; "
                f"expected {expected_output}"
            )

        print("First-case pipeline probe: PASS")

        del x_probe, y_probe, probe_image, probe_mask

    except Exception as exc:
        print(
            f"First-case probe failed: {type(exc).__name__}: {exc}"
        )
        print("\nFINAL STATUS: FAIL — DATA_PIPELINE_PROBE_FAILED")
        return 1

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    # -------------------------------------------------------------------------
    # 6. Full inference
    # -------------------------------------------------------------------------
    banner("6. FULL 100-ROW VALIDATION INFERENCE")

    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0

    total_start = time.perf_counter()

    for idx, row in cohort_df.iterrows():
        study_id = str(row.get("study_id", ""))
        series_id = str(row.get("series_id", ""))
        description = str(
            row.get(
                "series_description",
                row.get("description", ""),
            )
        )

        print(
            f"\n[{idx + 1:03d}/{len(cohort_df):03d}] "
            f"study={study_id} "
            f"series={series_id} "
            f"description={description}"
        )

        case_result: Dict[str, Any] = {
            "row_index": int(idx),
            "study_id": study_id,
            "series_id": series_id,
            "series_description": description,
            "status": "FAIL",
            "input_shape": None,
            "output_shape": None,
            "inference_seconds": None,
            "prediction_file": None,
            "prediction_foreground_voxels": None,
            "prediction_foreground_fraction": None,
            "target_foreground_voxels": None,
            "target_foreground_fraction": None,
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
            # Convert dict-like CSV row to pandas.Series explicitly.
            series_row = row.copy()

            # CRITICAL FIX:
            # Established Part 11 API is:
            #     load_tensor_case(part9, row)
            loaded = part11.load_tensor_case(
                part9,
                series_row,
            )

            if not isinstance(loaded, (tuple, list)):
                raise RuntimeError(
                    "Unexpected load_tensor_case return type."
                )

            if len(loaded) < 2:
                raise RuntimeError(
                    "load_tensor_case returned fewer than two values."
                )

            image = normalize_image(loaded[0])
            target_mask = normalize_mask(loaded[1])

            case_result["input_shape"] = str(
                tuple(image.shape)
            )

            x = image.unsqueeze(0).to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
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

            expected = (
                1,
                NUM_CLASSES,
                *FULL_SHAPE,
            )

            if tuple(logits.shape) != expected:
                raise RuntimeError(
                    f"Unexpected logits shape {tuple(logits.shape)}; "
                    f"expected {expected}"
                )

            counts = class_counts(prediction)

            dice_values = {
                c: dice_for_class(
                    prediction,
                    target_mask,
                    c,
                )
                for c in range(1, NUM_CLASSES)
            }

            total_voxels = int(prediction.numel())
            pred_fg = int((prediction > 0).sum())
            target_fg = int((target_mask > 0).sum())

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

            case_result.update(
                {
                    "status": "PASS",
                    "output_shape": str(tuple(logits.shape)),
                    "inference_seconds": round(
                        float(elapsed),
                        4,
                    ),
                    "prediction_file": str(output_path),
                    "prediction_foreground_voxels": pred_fg,
                    "prediction_foreground_fraction": (
                        pred_fg / total_voxels
                    ),
                    "target_foreground_voxels": target_fg,
                    "target_foreground_fraction": (
                        target_fg / total_voxels
                    ),
                    "class_0_voxels": counts[0],
                    "class_1_voxels": counts[1],
                    "class_2_voxels": counts[2],
                    "class_3_voxels": counts[3],
                    "class_4_voxels": counts[4],
                    "class_5_voxels": counts[5],
                    "class_1_dice_pseudomask": dice_values[1],
                    "class_2_dice_pseudomask": dice_values[2],
                    "class_3_dice_pseudomask": dice_values[3],
                    "class_4_dice_pseudomask": dice_values[4],
                    "class_5_dice_pseudomask": dice_values[5],
                }
            )

            successful += 1

            print(
                f"  Input      : {tuple(image.shape)}"
            )
            print(
                f"  Output     : {tuple(logits.shape)}"
            )
            print(
                f"  Time       : {elapsed:.3f} sec"
            )
            print(
                f"  Pred FG    : {pred_fg:,}"
            )
            print(
                f"  Target FG  : {target_fg:,}"
            )
            print(
                "  Class voxels: "
                + ", ".join(
                    f"C{c}={counts[c]:,}"
                    for c in range(NUM_CLASSES)
                )
            )
            print(
                "  Pseudo Dice: "
                + ", ".join(
                    (
                        f"C{c}={dice_values[c]:.4f}"
                        if not np.isnan(dice_values[c])
                        else f"C{c}=NaN"
                    )
                    for c in range(1, NUM_CLASSES)
                )
            )
            print(
                f"  Saved      : {output_name}"
            )

            del x, logits, prediction

        except Exception as exc:
            failed += 1
            case_result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"  INFERENCE FAIL — "
                f"{type(exc).__name__}: {exc}"
            )

        results.append(case_result)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_seconds = time.perf_counter() - total_start

    # -------------------------------------------------------------------------
    # 7. CSV
    # -------------------------------------------------------------------------
    banner("7. SAVE CASE RESULTS")

    if results:
        with CASE_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)

    print(f"Case CSV      : {CASE_CSV}")
    print(f"Predictions   : {PRED_DIR}")

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    banner("8. COHORT SUMMARY")

    passed = [
        r for r in results
        if r["status"] == "PASS"
    ]

    def numeric_values(key: str) -> List[float]:
        values = []
        for r in passed:
            value = r.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            values.append(value)
        return values

    class_summary = {}

    for c in range(1, NUM_CLASSES):
        key = f"class_{c}_dice_pseudomask"
        values = numeric_values(key)

        class_summary[
            f"class_{c}_{CLASS_NAMES[c]}_mean_pseudomask_dice"
        ] = (
            float(np.mean(values))
            if values
            else None
        )

        class_summary[
            f"class_{c}_{CLASS_NAMES[c]}_median_pseudomask_dice"
        ] = (
            float(np.median(values))
            if values
            else None
        )

    summary = {
        "part": 112,
        "status": (
            "PASS — FULL_VALIDATION_INFERENCE_COMPLETED"
            if successful == len(cohort_df)
            else (
                "PARTIAL — SOME_CASES_FAILED"
                if successful > 0
                else "FAIL — NO_CASES_PROCESSED"
            )
        ),
        "root_cause_fixed": [
            "Corrected Part 11 argument order to load_tensor_case(part9, row).",
            "Passed pandas.Series rows to the established loader.",
            "Preserved Part 11 preprocessing and mixed-DICOM compatibility behavior.",
        ],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_sha,
        "validation_cohort": str(VAL_COHORT),
        "cohort_rows": int(len(cohort_df)),
        "unique_studies": int(
            cohort_df["study_id"].nunique()
            if "study_id" in cohort_df
            else 0
        ),
        "successful_cases": successful,
        "failed_cases": failed,
        "total_inference_seconds": float(total_seconds),
        "mean_inference_seconds": (
            float(
                np.mean(
                    [
                        float(r["inference_seconds"])
                        for r in passed
                    ]
                )
            )
            if passed
            else None
        ),
        "architecture": {
            "name": "MONAI SwinUNETR",
            "in_channels": 1,
            "out_channels": 6,
            "feature_size": FEATURE_SIZE,
            "spatial_dims": 3,
            "use_checkpoint": False,
            "parameters": 4_078_116,
            "state_dict_keys": 159,
            "full_volume_shape": list(FULL_SHAPE),
        },
        "class_names": CLASS_NAMES,
        "prediction_directory": str(PRED_DIR),
        "case_results_csv": str(CASE_CSV),
        "pseudo_mask_metrics": {
            **class_summary,
            "note": (
                "These are development/pseudo-mask comparisons, not "
                "expert-ground-truth clinical segmentation metrics."
            ),
        },
        "failures": [
            {
                "row_index": r["row_index"],
                "study_id": r["study_id"],
                "series_id": r["series_id"],
                "error": r["error"],
            }
            for r in results
            if r["status"] != "PASS"
        ],
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 9. Report
    # -------------------------------------------------------------------------
    report = [
        "PART 112 — FIXED RSNA VALIDATION FULL-VOLUME INFERENCE",
        "=" * 72,
        "",
        f"Final status: {summary['status']}",
        "",
        "ROOT CAUSE FROM PART 111",
        "Part 11 defines load_tensor_case(part9, row), returning",
        "(image_tensor, mask_tensor, info).",
        "Part 111 passed the arguments in the reverse order.",
        "Part 111 also passed dictionary rows instead of pandas.Series rows.",
        "",
        "CORRECTION",
        "load_tensor_case(part9, pandas.Series(row))",
        "",
        "CHECKPOINT",
        f"SHA256: {actual_sha}",
        "",
        "ARCHITECTURE",
        "MONAI SwinUNETR",
        "in_channels=1",
        "out_channels=6",
        "feature_size=12",
        "spatial_dims=3",
        "use_checkpoint=False",
        f"volume={FULL_SHAPE}",
        "parameters=4,078,116",
        "state_dict_keys=159",
        "",
        "COHORT",
        f"Rows: {len(cohort_df)}",
        f"Unique studies: {summary['unique_studies']}",
        f"Successful: {successful}",
        f"Failed: {failed}",
        "",
        "RUNTIME",
        f"Device: {device}",
        f"Total seconds: {total_seconds:.3f}",
        f"Mean seconds/case: {summary['mean_inference_seconds']}",
        "",
        "SCIENTIFIC NOTE",
        "Pseudo-mask Dice values are development metrics against the "
        "project's existing pseudo-mask targets. They are not expert-"
        "ground-truth clinical validation metrics.",
        "",
        "OUTPUTS",
        str(CASE_CSV),
        str(PRED_DIR),
        str(SUMMARY_JSON),
    ]

    REPORT_TXT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 10. Final
    # -------------------------------------------------------------------------
    banner("PART 112 FINAL RESULT")

    print(summary["status"])
    print()
    print(
        f"Successfully processed : "
        f"{successful}/{len(cohort_df)}"
    )
    print(f"Failed                 : {failed}")
    print(
        f"Total inference time   : "
        f"{total_seconds:.3f} sec"
    )
    print()
    print("Outputs:")
    print(CASE_CSV)
    print(PRED_DIR)
    print(SUMMARY_JSON)
    print(REPORT_TXT)
    print("=" * 88)

    return 0 if successful == len(cohort_df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
