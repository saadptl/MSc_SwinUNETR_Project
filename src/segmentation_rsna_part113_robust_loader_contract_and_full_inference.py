"""
PART 113 — ROBUST PART 9 / PART 11 LOADER CONTRACT + FULL VALIDATION INFERENCE

Purpose
-------
Fix the Part 112 failure without changing the model, checkpoint, cohort, or
training pipeline.

Observed Part 112 error:
    AttributeError: 'Series' object has no attribute 'load_case'

This proves that the runtime Part 11 function in the user's current project
is receiving the pandas.Series in the position where it expects the Part 9
loader object. Rather than assuming the historical argument order, Part 113
inspects the LIVE function signature and selects the argument order from the
actual parameter names.

Supported runtime contracts:
    load_tensor_case(part9, row)
or
    load_tensor_case(row, part9)

The returned value may contain:
    image, mask, info

Part 113 preserves the existing Part 9/Part 11 preprocessing and performs
inference only.

NO TRAINING.
NO CHECKPOINT MODIFICATION.
NO src/model.py MODIFICATION.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


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
    / "rsna_part113_robust_loader_contract_and_full_inference"
)
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUTPUT_DIR / "part113_validation_case_inference.csv"
SUMMARY_JSON = REPORT_DIR / "part113_validation_inference_summary.json"
REPORT_TXT = REPORT_DIR / "part113_validation_inference_report.txt"

NUM_CLASSES = 6
FEATURE_SIZE = 12
FULL_SHAPE = (64, 96, 96)

EXPECTED_SHA256 = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)


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


def normalize_image(image: Any) -> torch.Tensor:
    image = torch.as_tensor(image).float().detach().cpu()

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.ndim != 4 or image.shape[0] != 1:
        raise RuntimeError(
            f"Expected image [1,D,H,W], got {tuple(image.shape)}"
        )

    if tuple(image.shape[-3:]) != FULL_SHAPE:
        raise RuntimeError(
            f"Expected image volume {FULL_SHAPE}, got "
            f"{tuple(image.shape[-3:])}"
        )

    return image.contiguous()


def normalize_mask(mask: Any) -> torch.Tensor:
    mask = torch.as_tensor(mask).long().detach().cpu()

    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim != 3:
        raise RuntimeError(
            f"Expected mask [D,H,W], got {tuple(mask.shape)}"
        )

    if tuple(mask.shape) != FULL_SHAPE:
        raise RuntimeError(
            f"Expected mask volume {FULL_SHAPE}, got {tuple(mask.shape)}"
        )

    labels = torch.unique(mask).tolist()
    if any(int(v) < 0 or int(v) >= NUM_CLASSES for v in labels):
        raise RuntimeError(f"Invalid mask labels: {labels}")

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

    p = int(pred.sum())
    t = int(true.sum())

    if p == 0 and t == 0:
        return float("nan")

    inter = int((pred & true).sum())
    return float((2.0 * inter) / (p + t + 1e-8))


def safe_name(value: Any) -> str:
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in str(value)
    )


def detect_loader_contract(part11) -> Dict[str, Any]:
    fn = getattr(part11, "load_tensor_case", None)
    if fn is None:
        raise RuntimeError("Part 11 load_tensor_case is missing.")

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    positional = [
        p for p in params
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    if len(positional) < 2:
        raise RuntimeError(
            f"Unexpected load_tensor_case signature: {sig}"
        )

    first = positional[0].name.lower()
    second = positional[1].name.lower()

    if (
        ("row" in first or "series" in first or "sample" in first)
        and ("part9" in second or "loader" in second)
    ):
        order = "row,part9"
    elif (
        ("part9" in first or "loader" in first)
        and ("row" in second or "series" in second or "sample" in second)
    ):
        order = "part9,row"
    else:
        order = "unknown"

    return {
        "signature": str(sig),
        "first_parameter": first,
        "second_parameter": second,
        "order": order,
    }


def call_loader(
    part9,
    part11,
    row: pd.Series,
    contract: Dict[str, Any],
):
    """
    Invoke the LIVE Part 11 function using its inspected parameter order.

    If the signature is ambiguous, safely try both orders. We do not modify
    Part 9 or Part 11.
    """
    order = contract["order"]

    if order == "part9,row":
        return part11.load_tensor_case(part9, row), "part9,row"

    if order == "row,part9":
        return part11.load_tensor_case(row, part9), "row,part9"

    attempts = [
        ("part9,row", lambda: part11.load_tensor_case(part9, row)),
        ("row,part9", lambda: part11.load_tensor_case(row, part9)),
    ]

    errors = []

    for name, fn in attempts:
        try:
            return fn(), name
        except Exception as exc:
            errors.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Both possible loader argument orders failed:\n"
        + "\n".join(errors)
    )


def unpack_loaded(loaded: Any):
    if not isinstance(loaded, (tuple, list)):
        raise RuntimeError(
            f"Unexpected load_tensor_case return type: "
            f"{type(loaded).__name__}"
        )

    if len(loaded) < 2:
        raise RuntimeError(
            f"load_tensor_case returned {len(loaded)} values; "
            "expected at least image and mask."
        )

    return loaded[0], loaded[1], (
        loaded[2] if len(loaded) >= 3 else {}
    )


def main() -> int:
    banner("PART 113 — ROBUST LOADER CONTRACT + FULL VALIDATION INFERENCE")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    banner("1. INPUT VALIDATION")

    required = {
        "validation cohort": VAL_COHORT,
        "final checkpoint": CHECKPOINT,
        "Part 9 loader": PART9_PATH,
        "Part 11 loader": PART11_PATH,
    }

    for name, path in required.items():
        if not path.exists():
            print(f"{name:<25}: MISSING")
            print("\nFINAL STATUS: FAIL — REQUIRED_INPUT_MISSING")
            return 1
        print(f"{name:<25}: FOUND")

    actual_sha = sha256_file(CHECKPOINT)
    print(f"\nCheckpoint SHA256 : {actual_sha}")
    print(f"Expected SHA256   : {EXPECTED_SHA256}")

    if actual_sha != EXPECTED_SHA256:
        print("SHA256 validation : FAIL")
        print("\nFINAL STATUS: FAIL — CHECKPOINT_SHA_MISMATCH")
        return 1

    print("SHA256 validation : PASS")

    banner("2. VALIDATION COHORT")

    cohort_df = pd.read_csv(VAL_COHORT)

    if cohort_df.empty:
        print("\nFINAL STATUS: FAIL — EMPTY_VALIDATION_COHORT")
        return 1

    print(f"Cohort rows       : {len(cohort_df)}")

    if "study_id" in cohort_df.columns:
        print(f"Unique studies   : {cohort_df['study_id'].nunique()}")

    banner("3. LIVE PART 9 / PART 11 CONTRACT")

    try:
        part9 = import_from_path("part113_part9", PART9_PATH)
        part11 = import_from_path("part113_part11", PART11_PATH)
    except Exception as exc:
        print(f"Import failed: {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — PIPELINE_IMPORT_FAILED")
        return 1

    if not hasattr(part9, "load_case"):
        print("Part 9 load_case : MISSING")
        return 1

    if not hasattr(part11, "load_tensor_case"):
        print("Part 11 load_tensor_case : MISSING")
        return 1

    contract = detect_loader_contract(part11)

    print(f"Live signature : {contract['signature']}")
    print(f"Parameter 1    : {contract['first_parameter']}")
    print(f"Parameter 2    : {contract['second_parameter']}")
    print(f"Detected order : {contract['order']}")

    if contract["order"] == "unknown":
        print(
            "Signature is ambiguous; Part 113 will test both valid "
            "historical argument orders on the first case."
        )
    else:
        print("Contract detection: PASS")

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

        params = sum(p.numel() for p in model.parameters())
        keys = len(model.state_dict())

        print("Architecture    : MONAI SwinUNETR")
        print("in_channels     : 1")
        print("out_channels    : 6")
        print("feature_size    : 12")
        print("spatial_dims    : 3")
        print("use_checkpoint  : False")
        print(f"Parameters      : {params:,}")
        print(f"State keys      : {keys}")

        if params != 4_078_116 or keys != 159:
            raise RuntimeError(
                f"Unexpected canonical model fingerprint: "
                f"params={params}, keys={keys}"
            )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        state = extract_state_dict(checkpoint)
        state = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in state.items()
        }

        model.load_state_dict(state, strict=True)
        print("Strict checkpoint load: PASS")

    except Exception as exc:
        print(f"Model setup failed: {type(exc).__name__}: {exc}")
        print("\nFINAL STATUS: FAIL — MODEL_SETUP_FAILED")
        return 1

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    print(f"Inference device: {device}")

    banner("5. FIRST-CASE DATA PIPELINE PROBE")

    first_row = cohort_df.iloc[0].copy()

    print(f"Study ID : {first_row.get('study_id', '')}")
    print(f"Series ID: {first_row.get('series_id', '')}")
    print(
        f"Series   : "
        f"{first_row.get('series_description', first_row.get('description', ''))}"
    )

    try:
        loaded, used_order = call_loader(
            part9,
            part11,
            first_row,
            contract,
        )

        print(f"Loader invocation used: {used_order}")

        image_raw, mask_raw, info = unpack_loaded(loaded)

        image = normalize_image(image_raw)
        mask = normalize_mask(mask_raw)

        print(f"Image tensor : {tuple(image.shape)}")
        print(f"Mask tensor  : {tuple(mask.shape)}")
        print(
            f"Image range  : "
            f"{float(image.min()):.5f} → {float(image.max()):.5f}"
        )
        print(
            f"Mask labels  : "
            f"{sorted(int(x) for x in torch.unique(mask))}"
        )

        x = image.unsqueeze(0).to(device)

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    y = model(x)
            else:
                y = model(x)

        expected = (1, NUM_CLASSES, *FULL_SHAPE)

        print(f"Forward output: {tuple(y.shape)}")

        if tuple(y.shape) != expected:
            raise RuntimeError(
                f"Unexpected forward output {tuple(y.shape)}; "
                f"expected {expected}"
            )

        print("First-case data pipeline + forward probe: PASS")

        del x, y, image, mask, info, image_raw, mask_raw

    except Exception as exc:
        print(
            f"First-case probe failed: "
            f"{type(exc).__name__}: {exc}"
        )
        print("\nFINAL STATUS: FAIL — DATA_PIPELINE_PROBE_FAILED")
        return 1

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    banner("6. FULL 100-ROW VALIDATION INFERENCE")

    results: List[Dict[str, Any]] = []
    successful = 0
    failed = 0
    total_start = time.perf_counter()

    for pos, (_, row) in enumerate(
        cohort_df.iterrows(),
        start=1,
    ):
        row = row.copy()

        study_id = str(row.get("study_id", ""))
        series_id = str(row.get("series_id", ""))
        description = str(
            row.get(
                "series_description",
                row.get("description", ""),
            )
        )

        print(
            f"\n[{pos:03d}/{len(cohort_df):03d}] "
            f"study={study_id} "
            f"series={series_id} "
            f"description={description}"
        )

        result: Dict[str, Any] = {
            "row_position": pos,
            "study_id": study_id,
            "series_id": series_id,
            "series_description": description,
            "status": "FAIL",
            "loader_order": "",
            "input_shape": "",
            "output_shape": "",
            "inference_seconds": None,
            "prediction_file": "",
            "prediction_foreground_voxels": None,
            "target_foreground_voxels": None,
            "prediction_foreground_fraction": None,
            "target_foreground_fraction": None,
            "class_1_dice_pseudomask": None,
            "class_2_dice_pseudomask": None,
            "class_3_dice_pseudomask": None,
            "class_4_dice_pseudomask": None,
            "class_5_dice_pseudomask": None,
            "error": "",
        }

        try:
            loaded, used_order = call_loader(
                part9,
                part11,
                row,
                contract,
            )

            result["loader_order"] = used_order

            image_raw, mask_raw, _ = unpack_loaded(loaded)

            image = normalize_image(image_raw)
            target = normalize_mask(mask_raw)

            result["input_shape"] = str(tuple(image.shape))

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

            result["inference_seconds"] = elapsed
            result["output_shape"] = str(tuple(logits.shape))

            expected = (1, NUM_CLASSES, *FULL_SHAPE)

            if tuple(logits.shape) != expected:
                raise RuntimeError(
                    f"Unexpected logits shape {tuple(logits.shape)}; "
                    f"expected {expected}"
                )

            pred_fg = int((prediction > 0).sum())
            target_fg = int((target > 0).sum())
            voxels = int(prediction.numel())

            result["prediction_foreground_voxels"] = pred_fg
            result["target_foreground_voxels"] = target_fg
            result["prediction_foreground_fraction"] = (
                pred_fg / voxels
            )
            result["target_foreground_fraction"] = (
                target_fg / voxels
            )

            for c in range(1, NUM_CLASSES):
                result[f"class_{c}_dice_pseudomask"] = (
                    dice_for_class(prediction, target, c)
                )

            filename = (
                f"{pos:03d}_study_{safe_name(study_id)}"
                f"_series_{safe_name(series_id)}.npz"
            )

            pred_path = PRED_DIR / filename

            np.savez_compressed(
                pred_path,
                prediction=prediction.numpy().astype(np.uint8),
            )

            result["prediction_file"] = str(
                pred_path.relative_to(PROJECT_ROOT)
            )
            result["status"] = "PASS"

            successful += 1

            mean_dice = float(
                np.nanmean(
                    [
                        result[f"class_{c}_dice_pseudomask"]
                        for c in range(1, NUM_CLASSES)
                    ]
                )
            )

            print(
                f"  PASS — time={elapsed:.3f}s "
                f"pred_fg={pred_fg} "
                f"target_fg={target_fg} "
                f"mean_class_dice={mean_dice:.4f}"
            )

            del x, logits, prediction, image, target

        except Exception as exc:
            failed += 1
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"  INFERENCE FAIL — "
                f"{type(exc).__name__}: {exc}"
            )

        results.append(result)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    elapsed_total = time.perf_counter() - total_start

    result_df = pd.DataFrame(results)
    result_df.to_csv(CASE_CSV, index=False)

    banner("7. FINAL SUMMARY")

    print(f"Processed successfully : {successful}/{len(cohort_df)}")
    print(f"Failed                 : {failed}")
    print(f"Total inference time   : {elapsed_total:.2f} sec")
    print(f"Case results CSV       : {CASE_CSV}")
    print(f"Prediction directory   : {PRED_DIR}")

    summary = {
        "part": 113,
        "status": (
            "PASS — FULL_VALIDATION_INFERENCE_COMPLETED"
            if successful == len(cohort_df)
            else (
                "PARTIAL — VALIDATION_INFERENCE_COMPLETED_WITH_FAILURES"
                if successful > 0
                else "FAIL — NO_VALIDATION_CASES_PROCESSED"
            )
        ),
        "project_root": str(PROJECT_ROOT),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_sha,
        "expected_checkpoint_sha256": EXPECTED_SHA256,
        "cohort_rows": int(len(cohort_df)),
        "successful_cases": int(successful),
        "failed_cases": int(failed),
        "total_seconds": float(elapsed_total),
        "device": str(device),
        "model_parameters": 4_078_116,
        "model_state_keys": 159,
        "full_shape": list(FULL_SHAPE),
        "loader_contract": contract,
        "outputs": {
            "case_csv": str(CASE_CSV),
            "prediction_directory": str(PRED_DIR),
        },
        "scientific_note": (
            "Per-class Dice values compare predictions against the existing "
            "project pseudo-masks/development labels. They are not clinical "
            "expert-ground-truth performance measures."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "PART 113 — ROBUST LOADER CONTRACT + FULL VALIDATION INFERENCE",
        "",
        f"Status: {summary['status']}",
        f"Cohort rows: {len(cohort_df)}",
        f"Successful: {successful}",
        f"Failed: {failed}",
        f"Device: {device}",
        f"Checkpoint SHA256: {actual_sha}",
        "",
        "LIVE LOADER CONTRACT",
        f"Signature: {contract['signature']}",
        f"Detected order: {contract['order']}",
        "",
        "OUTPUTS",
        f"Case CSV: {CASE_CSV}",
        f"Prediction directory: {PRED_DIR}",
        f"Summary JSON: {SUMMARY_JSON}",
        "",
        "Scientific note:",
        "Dice values are against project pseudo-masks/development labels,",
        "not clinical expert-ground-truth annotations.",
    ]

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 113 FINAL RESULT")

    if successful == len(cohort_df):
        print("PASS — FULL VALIDATION INFERENCE COMPLETED")
        return 0

    if successful > 0:
        print("PARTIAL — VALIDATION INFERENCE COMPLETED WITH FAILURES")
        return 2

    print("FAIL — NO VALIDATION CASES PROCESSED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
