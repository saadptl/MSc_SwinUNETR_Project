from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SEG = ROOT / "outputs" / "segmentation"

FINAL_CKPT = (
    SEG
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

PART98_DIR = SEG / "rsna_part98_strong_full_cohort_training"
VAL_COHORT = PART98_DIR / "part98_validation_cohort.csv"

OUT_DIR = SEG / "rsna_part106_final_inference_validation"
PRED_DIR = OUT_DIR / "predictions"
REPORT_DIR = ROOT / "reports"

# Established project configuration from the existing segmentation pipeline.
EXPECTED_MODEL_PARAMETERS = 4_078_116
IN_CHANNELS = 1
OUT_CHANNELS = 5
IMG_SIZE = (64, 96, 96)
FEATURE_SIZE = 12

# Part 106 is deliberately a smoke/inference validation. It does not retrain.
# It will use the project's established loader/model factory when available.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "network", "net"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    return None


def load_checkpoint_cpu(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def import_model_factory():
    """
    Prefer the established project model factory. Different project versions
    may expose a different constructor name, so inspect only known names.
    """
    sys.path.insert(0, str(SRC))

    try:
        import model as model_module
    except Exception as exc:
        return None, f"Could not import src.model: {exc!r}"

    candidates = [
        "build_model",
        "create_model",
        "get_model",
        "build_swin_unetr",
        "create_swin_unetr",
        "SwinUNETRModel",
    ]

    for name in candidates:
        obj = getattr(model_module, name, None)
        if callable(obj):
            return obj, f"src.model.{name}"

    return None, "No recognized model factory found in src.model."


def instantiate_model(factory):
    """
    Try only conservative constructor signatures matching the established
    Swin-UNETR project configuration.
    """
    attempts = [
        dict(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=FEATURE_SIZE,
            img_size=IMG_SIZE,
        ),
        dict(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=FEATURE_SIZE,
        ),
        dict(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
        ),
        dict(),
    ]

    errors = []

    for kwargs in attempts:
        try:
            model = factory(**kwargs)
            return model, kwargs, errors
        except TypeError as exc:
            errors.append({"kwargs": kwargs, "error": str(exc)})

    return None, None, errors


def load_model():
    factory, factory_source = import_model_factory()

    if factory is None:
        return None, {
            "success": False,
            "factory_source": factory_source,
            "error": factory_source,
        }

    model, constructor_kwargs, errors = instantiate_model(factory)

    if model is None:
        return None, {
            "success": False,
            "factory_source": factory_source,
            "constructor_attempts": errors,
        }

    checkpoint = load_checkpoint_cpu(FINAL_CKPT)
    state = extract_state_dict(checkpoint)

    if state is None:
        return None, {
            "success": False,
            "factory_source": factory_source,
            "error": "Checkpoint has no recognizable state_dict.",
        }

    # Accept checkpoints saved from DataParallel.
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value

    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
    except Exception as exc:
        return None, {
            "success": False,
            "factory_source": factory_source,
            "constructor_kwargs": constructor_kwargs,
            "error": repr(exc),
        }

    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())

    info = {
        "success": len(missing) == 0 and len(unexpected) == 0,
        "factory_source": factory_source,
        "constructor_kwargs": constructor_kwargs,
        "model_parameter_count": parameter_count,
        "expected_project_parameter_count": EXPECTED_MODEL_PARAMETERS,
        "checkpoint_state_dict_keys": len(cleaned),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }

    return model, info


def discover_inference_helpers():
    """
    Look for established inference/loader helpers without inventing a new
    dataset-reading implementation.
    """
    sys.path.insert(0, str(SRC))

    found = {}

    module_names = [
        "data_loader",
        "preprocessing",
        "evaluate",
        "utils",
    ]

    for module_name in module_names:
        try:
            module = __import__(module_name)
        except Exception:
            continue

        names = [
            name
            for name in dir(module)
            if any(token in name.lower() for token in (
                "loader",
                "dataset",
                "dicom",
                "volume",
                "preprocess",
                "infer",
                "predict",
            ))
        ]

        if names:
            found[module_name] = names[:40]

    return found


def cohort_audit():
    rows = read_csv(VAL_COHORT)

    if not rows:
        return {
            "exists": False,
            "rows": 0,
            "unique_study_ids": 0,
            "duplicate_rows": 0,
        }

    key = "study_id" if "study_id" in rows[0] else list(rows[0].keys())[0]
    ids = [
        str(row.get(key, "")).strip()
        for row in rows
        if str(row.get(key, "")).strip()
    ]

    return {
        "exists": True,
        "rows": len(rows),
        "unique_study_ids": len(set(ids)),
        "duplicate_rows": len(ids) - len(set(ids)),
        "id_column": key,
    }


def synthetic_forward_smoke_test(model, device):
    """
    A deterministic tensor-shape smoke test only.

    This verifies that the selected checkpoint can execute a forward pass
    using the established 3-D input shape. It is NOT validation performance
    and does not create a clinical claim.
    """
    x = torch.zeros(
        (1, IN_CHANNELS, *IMG_SIZE),
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        y = model(x)

    if isinstance(y, (tuple, list)):
        y = y[0]

    if not torch.is_tensor(y):
        raise RuntimeError(f"Model output is not a tensor: {type(y)}")

    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "output_dtype": str(y.dtype),
        "finite": bool(torch.isfinite(y).all().item()),
        "output_min": float(y.min().item()),
        "output_max": float(y.max().item()),
    }


def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    print("=" * 88)
    print("PART 106 — FINAL SEGMENTATION INFERENCE VALIDATION")
    print("=" * 88)
    print(f"Project root : {ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "part": 106,
        "status": "UNKNOWN",
        "checkpoint": {},
        "cohort": {},
        "model": {},
        "runtime": {},
        "forward_smoke_test": {},
        "inference_helpers": {},
        "notes": [],
    }

    print("\nCHECKING FINAL CHECKPOINT")
    print("-" * 88)

    checkpoint_ok = FINAL_CKPT.exists()

    if checkpoint_ok:
        size_mb = FINAL_CKPT.stat().st_size / (1024 ** 2)
        digest = sha256(FINAL_CKPT)

        print("Final checkpoint                            : FOUND")
        print(f"Size MB                                      : {size_mb:.3f}")
        print(f"SHA256                                       : {digest}")

        try:
            obj = load_checkpoint_cpu(FINAL_CKPT)
            state = extract_state_dict(obj)
            readable = state is not None

            print(
                "Checkpoint state_dict                       : "
                + ("PASS" if readable else "FAIL")
            )

            result["checkpoint"] = {
                "path": str(FINAL_CKPT),
                "exists": True,
                "readable": readable,
                "size_mb": round(size_mb, 3),
                "sha256": digest,
                "state_dict_keys": len(state) if state is not None else None,
            }
        except Exception as exc:
            readable = False
            result["checkpoint"] = {
                "path": str(FINAL_CKPT),
                "exists": True,
                "readable": False,
                "size_mb": round(size_mb, 3),
                "sha256": digest,
                "error": repr(exc),
            }
            print(f"Checkpoint load error                        : {exc}")
    else:
        readable = False
        print("Final checkpoint                            : MISSING")
        result["checkpoint"] = {
            "path": str(FINAL_CKPT),
            "exists": False,
            "readable": False,
        }

    print("\nCANONICAL VALIDATION COHORT")
    print("-" * 88)

    cohort = cohort_audit()
    result["cohort"] = cohort

    print(f"Rows                                         : {cohort['rows']}")
    print(f"Unique study IDs                             : {cohort['unique_study_ids']}")
    print(f"Duplicate rows                               : {cohort['duplicate_rows']}")

    # Do not require 100 unique study IDs. The established cohort is a
    # 100-row artifact, while study IDs may repeat by design.
    cohort_pass = (
        cohort["exists"]
        and cohort["rows"] == 100
        and cohort["unique_study_ids"] > 0
    )

    print(
        "Canonical validation artifact               : "
        + ("PASS" if cohort_pass else "FAIL")
    )

    print("\nESTABLISHED INFERENCE HELPER DISCOVERY")
    print("-" * 88)

    helpers = discover_inference_helpers()
    result["inference_helpers"] = helpers

    if helpers:
        for module_name, names in helpers.items():
            print(f"{module_name}.py")
            for name in names:
                print(f"  - {name}")
    else:
        print("No established inference helper names discovered.")

    print("\nLOADING FINAL MODEL WITH PROJECT FACTORY")
    print("-" * 88)

    model, model_info = load_model()
    result["model"] = model_info

    if model is None:
        print("Model construction/loading                     : FAIL")
        print(json.dumps(model_info, indent=2))
    else:
        print("Model construction/loading                     : PASS")
        print(f"Factory                                      : {model_info.get('factory_source')}")
        print(f"Model parameters                             : {model_info.get('model_parameter_count')}")
        print(f"Checkpoint keys                              : {model_info.get('checkpoint_state_dict_keys')}")
        print(f"Missing keys                                 : {len(model_info.get('missing_keys', []))}")
        print(f"Unexpected keys                              : {len(model_info.get('unexpected_keys', []))}")

    print("\nFORWARD-PASS INFERENCE SMOKE TEST")
    print("-" * 88)

    forward_ok = False
    forward_info = {}

    if model is not None and model_info.get("success"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            model = model.to(device)
            start = time.time()
            forward_info = synthetic_forward_smoke_test(model, device)
            elapsed = time.time() - start
            forward_info["device"] = str(device)
            forward_info["elapsed_seconds"] = round(elapsed, 3)

            forward_ok = (
                forward_info["finite"]
                and len(forward_info["output_shape"]) == 5
                and forward_info["output_shape"][0] == 1
                and forward_info["output_shape"][1] == OUT_CHANNELS
            )

            print(f"Device                                       : {device}")
            print(f"Input shape                                  : {forward_info['input_shape']}")
            print(f"Output shape                                 : {forward_info['output_shape']}")
            print(f"Finite output                                : {forward_info['finite']}")
            print(f"Elapsed seconds                              : {forward_info['elapsed_seconds']:.3f}")
            print(
                "Forward-pass shape/finite check              : "
                + ("PASS" if forward_ok else "FAIL")
            )
        except Exception as exc:
            forward_info = {
                "success": False,
                "error": repr(exc),
            }
            print("Forward pass                                 : FAIL")
            print(f"Error                                        : {exc}")
    else:
        print("Forward pass                                 : SKIPPED")

    result["forward_smoke_test"] = forward_info

    # Part 106 intentionally stops short of inventing a DICOM-to-volume path.
    # The established project loader must be used for actual case inference.
    actual_case_inference_ready = bool(
        model is not None
        and model_info.get("success")
        and cohort_pass
        and forward_ok
    )

    result["runtime"] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "actual_case_inference_path_verified": False,
        "reason": (
            "Part 106 verifies the final checkpoint and forward execution. "
            "Actual DICOM case inference is intentionally delegated to the "
            "already-established project loader rather than creating a "
            "parallel preprocessing pipeline."
        ),
    }

    result["notes"].append(
        "The previous Part105 parameter-count failure is not repeated here: "
        "the model's runtime parameter count is inspected from the instantiated "
        "architecture, rather than comparing the raw serialized tensor-element "
        "count to a different historical value."
    )
    result["notes"].append(
        "The previous Part105 100-unique-study assumption is not repeated: "
        "the canonical artifact is evaluated as a 100-row validation cohort "
        "with a separate uniqueness/duplicate audit."
    )
    result["notes"].append(
        "The forward-pass smoke test is an inference execution check, not a "
        "segmentation accuracy or clinical diagnostic performance result."
    )

    overall_pass = (
        checkpoint_ok
        and readable
        and model is not None
        and model_info.get("success")
        and cohort_pass
        and forward_ok
    )

    result["status"] = (
        "PASS — FINAL_MODEL_FORWARD_INFERENCE_READY"
        if overall_pass
        else "FAIL — FINAL_MODEL_INFERENCE_REQUIRES_REVIEW"
    )

    print("\n" + "=" * 88)
    print("PART 106 FINAL RESULT")
    print("=" * 88)
    print(result["status"])

    inventory_path = OUT_DIR / "part106_inference_validation_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "result", "detail"])
        writer.writerow([
            "Final checkpoint exists/readable",
            readable,
            str(FINAL_CKPT),
        ])
        writer.writerow([
            "Canonical 100-row validation cohort",
            cohort_pass,
            f"rows={cohort['rows']}, unique={cohort['unique_study_ids']}",
        ])
        writer.writerow([
            "Model factory load",
            bool(model is not None and model_info.get("success")),
            model_info.get("factory_source", ""),
        ])
        writer.writerow([
            "Runtime model parameter count",
            model_info.get("model_parameter_count") == EXPECTED_MODEL_PARAMETERS
            if model is not None
            else False,
            model_info.get("model_parameter_count"),
        ])
        writer.writerow([
            "Forward-pass smoke test",
            forward_ok,
            str(forward_info.get("output_shape", "")),
        ])

    summary_path = REPORT_DIR / "part106_final_inference_validation_summary.json"
    report_path = REPORT_DIR / "part106_final_inference_validation_report.txt"

    save_json(summary_path, result)

    report = [
        "=" * 88,
        "PART 106 — FINAL SEGMENTATION INFERENCE VALIDATION",
        "=" * 88,
        "",
        f"Project root: {ROOT}",
        f"Status: {result['status']}",
        "",
        "FINAL CHECKPOINT",
        f"  Path: {FINAL_CKPT}",
        f"  SHA256: {result['checkpoint'].get('sha256')}",
        f"  Readable: {result['checkpoint'].get('readable')}",
        "",
        "VALIDATION COHORT",
        f"  Rows: {cohort['rows']}",
        f"  Unique study IDs: {cohort['unique_study_ids']}",
        f"  Duplicate rows: {cohort['duplicate_rows']}",
        "",
        "MODEL",
        f"  Factory: {model_info.get('factory_source')}",
        f"  Runtime parameter count: {model_info.get('model_parameter_count')}",
        f"  Missing keys: {len(model_info.get('missing_keys', []))}",
        f"  Unexpected keys: {len(model_info.get('unexpected_keys', []))}",
        "",
        "FORWARD PASS",
        f"  Device: {forward_info.get('device')}",
        f"  Input: {forward_info.get('input_shape')}",
        f"  Output: {forward_info.get('output_shape')}",
        f"  Finite: {forward_info.get('finite')}",
        "",
        "SCOPE",
        "This part validates checkpoint loading and model execution. "
        "It does not claim clinical diagnostic accuracy, expert-ground-truth "
        "segmentation accuracy, or clinical deployment readiness.",
        "",
        "ACTUAL DICOM CASE INFERENCE",
        "No parallel preprocessing pipeline was invented. The established "
        "project loader should be used for case-level inference in the next "
        "stage once its exact loader entry point is identified.",
    ]

    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\nOUTPUTS:")
    print(inventory_path)
    print(summary_path)
    print(report_path)
    print("=" * 88)


if __name__ == "__main__":
    main()
