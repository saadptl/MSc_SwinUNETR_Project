"""
PART 109 — DIRECT Swin-UNETR CHECKPOINT LOAD & FORWARD VALIDATION

Purpose
-------
Part 108 found a critical result: the historical project scripts construct
MONAI SwinUNETR with:
    in_channels=1
    out_channels=6
    feature_size=12
and the resulting model has exactly:
    4,078,116 parameters
    159 state-dict keys

The Part 108 candidate logic nevertheless reported EXACT MATCH=False, so this
Part 109 removes that ambiguity and performs the direct test against the exact
architecture.

This script:
- loads the Part 104-selected checkpoint read-only;
- imports MONAI SwinUNETR directly;
- constructs exactly the known 1/6/12 architecture;
- compares every checkpoint key and tensor shape;
- identifies any non-tensor/state-dict edge cases;
- performs strict=True checkpoint loading;
- performs a CPU forward pass;
- optionally performs a CUDA forward pass if enough GPU memory is available;
- reports output shape, logits statistics and predicted class distribution.

No training, source modification, package installation, or checkpoint overwrite.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part109_direct_swinunetr_checkpoint_load_validation"
)
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "part109_key_shape_validation.csv"
JSON_PATH = REPORT_DIR / "part109_direct_swinunetr_validation_summary.json"
TXT_PATH = REPORT_DIR / "part109_direct_swinunetr_validation_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj: Any):
    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint root is {type(obj).__name__}, expected dict.")

    for name in (
        "model_state_dict",
        "state_dict",
        "model",
        "net",
        "network",
        "weights",
        "model_weights",
    ):
        candidate = obj.get(name)
        if isinstance(candidate, dict):
            tensor_items = {
                k: v for k, v in candidate.items()
                if isinstance(k, str) and torch.is_tensor(v)
            }
            if tensor_items:
                return name, tensor_items

    tensor_items = {
        k: v for k, v in obj.items()
        if isinstance(k, str) and torch.is_tensor(v)
    }
    if tensor_items:
        return "root", tensor_items

    raise RuntimeError("No tensor state_dict found.")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def tensor_numel(sd):
    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))


def json_safe(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "numel": int(x.numel()),
        }
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def forward_info(output):
    if isinstance(output, (tuple, list)):
        primary = output[0]
        container = type(output).__name__
    elif isinstance(output, dict):
        primary = None
        for key in ("out", "logits", "pred", "seg"):
            if key in output and torch.is_tensor(output[key]):
                primary = output[key]
                break
        if primary is None:
            raise RuntimeError("Dictionary output has no recognized tensor.")
        container = "dict"
    else:
        primary = output
        container = type(output).__name__

    if not torch.is_tensor(primary):
        raise RuntimeError(f"Output is not a tensor: {type(primary).__name__}")

    with torch.no_grad():
        probs = torch.softmax(primary.float(), dim=1)
        pred = probs.argmax(dim=1)

    unique, counts = torch.unique(pred, return_counts=True)
    distribution = {
        str(int(k)): int(v)
        for k, v in zip(unique.cpu(), counts.cpu())
    }

    return {
        "container": container,
        "shape": list(primary.shape),
        "dtype": str(primary.dtype),
        "min": float(primary.float().min().item()),
        "max": float(primary.float().max().item()),
        "mean": float(primary.float().mean().item()),
        "std": float(primary.float().std().item()),
        "predicted_class_voxel_counts": distribution,
        "channel_count": int(primary.shape[1]) if primary.ndim >= 2 else None,
    }


def main():
    print("=" * 88)
    print("PART 109 — DIRECT Swin-UNETR CHECKPOINT LOAD & FORWARD VALIDATION")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print()

    summary = {
        "part": 109,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": str(CHECKPOINT),
    }

    print("1. FINAL CHECKPOINT")
    print("-" * 88)
    if not CHECKPOINT.exists():
        print("Checkpoint : MISSING")
        print("FINAL: FAIL — CHECKPOINT_MISSING")
        return 1

    sha = sha256_file(CHECKPOINT)
    size_mb = CHECKPOINT.stat().st_size / (1024 * 1024)
    print("Checkpoint : FOUND")
    print(f"Size MB    : {size_mb:.3f}")
    print(f"SHA256     : {sha}")

    ckpt = load_checkpoint(CHECKPOINT)
    source, ckpt_sd = extract_state_dict(ckpt)

    metadata = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if k not in {
                "model_state_dict", "state_dict", "model", "net",
                "network", "weights", "model_weights",
                "optimizer_state_dict", "scaler_state_dict",
            } and not torch.is_tensor(v):
                metadata[k] = json_safe(v)

    print(f"State source       : {source}")
    print(f"Checkpoint keys    : {len(ckpt_sd)}")
    print(f"Tensor elements    : {tensor_numel(ckpt_sd):,}")

    for key in ("epoch", "feature_size", "num_classes", "full_shape", "crop_shape"):
        if key in metadata:
            print(f"{key:<19}: {metadata[key]}")

    summary["checkpoint_sha256"] = sha
    summary["checkpoint_size_mb"] = size_mb
    summary["state_source"] = source
    summary["checkpoint_key_count"] = len(ckpt_sd)
    summary["checkpoint_tensor_numel"] = tensor_numel(ckpt_sd)
    summary["metadata"] = metadata

    print("\n2. DIRECT MONAI SwinUNETR IMPORT")
    print("-" * 88)
    try:
        import monai
        from monai.networks.nets import SwinUNETR

        print(f"MONAI version : {getattr(monai, '__version__', 'unknown')}")
        print(f"SwinUNETR     : {SwinUNETR}")
        print("Import        : PASS")

        summary["monai_version"] = getattr(monai, "__version__", None)
        summary["swinunetr_import"] = "PASS"
    except Exception as e:
        print(f"Import : FAIL — {type(e).__name__}: {e}")
        summary["swinunetr_import"] = f"FAIL: {type(e).__name__}: {e}"
        summary["final_status"] = "FAIL — MONAI_SWINUNETR_UNAVAILABLE"
        JSON_PATH.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(
            "PART 109\n\nMONAI SwinUNETR import failed.\n" + str(e),
            encoding="utf-8",
        )
        print("\nFINAL: FAIL — MONAI_SWINUNETR_UNAVAILABLE")
        return 1

    print("\n3. EXACT ARCHITECTURE CONSTRUCTION")
    print("-" * 88)

    # These values are independently confirmed by the checkpoint metadata and
    # historical project scripts discovered by Part 108.
    kwargs = {
        "in_channels": 1,
        "out_channels": 6,
        "feature_size": 12,
        "spatial_dims": 3,
        "use_checkpoint": False,
    }

    try:
        model = SwinUNETR(**kwargs)
    except TypeError:
        # Compatibility fallback for MONAI versions without spatial_dims.
        kwargs.pop("spatial_dims", None)
        model = SwinUNETR(**kwargs)

    model = model.cpu().eval()
    model_sd = model.state_dict()

    print("Configuration:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print(f"Model parameters  : {parameter_count(model):,}")
    print(f"Model state keys  : {len(model_sd)}")
    print(f"Model tensor nums : {tensor_numel(model_sd):,}")

    summary["model_kwargs"] = kwargs
    summary["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
    summary["model_parameters"] = parameter_count(model)
    summary["model_state_keys"] = len(model_sd)
    summary["model_tensor_numel"] = tensor_numel(model_sd)

    print("\n4. COMPLETE KEY/SHAPE COMPARISON")
    print("-" * 88)

    ckpt_keys = set(ckpt_sd)
    model_keys = set(model_sd)
    common = ckpt_keys & model_keys
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    rows = []
    shape_mismatches = []
    shape_matches = []

    for key in sorted(ckpt_keys | model_keys):
        c = ckpt_sd.get(key)
        m = model_sd.get(key)

        if c is None:
            status = "MISSING_IN_CHECKPOINT"
        elif m is None:
            status = "UNEXPECTED_IN_CHECKPOINT"
        elif tuple(c.shape) != tuple(m.shape):
            status = "SHAPE_MISMATCH"
            shape_mismatches.append(key)
        else:
            status = "MATCH"
            shape_matches.append(key)

        rows.append({
            "key": key,
            "status": status,
            "checkpoint_shape": list(c.shape) if c is not None else "",
            "model_shape": list(m.shape) if m is not None else "",
            "checkpoint_numel": int(c.numel()) if c is not None else "",
            "model_numel": int(m.numel()) if m is not None else "",
            "checkpoint_dtype": str(c.dtype) if c is not None else "",
            "model_dtype": str(m.dtype) if m is not None else "",
        })

    print(f"Checkpoint keys      : {len(ckpt_keys)}")
    print(f"Model keys           : {len(model_keys)}")
    print(f"Common keys          : {len(common)}")
    print(f"Shape matches        : {len(shape_matches)}")
    print(f"Shape mismatches     : {len(shape_mismatches)}")
    print(f"Missing keys         : {len(missing)}")
    print(f"Unexpected keys      : {len(unexpected)}")

    print("\nCheckpoint-only keys:")
    for k in sorted(unexpected)[:30]:
        print("  " + k)

    print("\nModel-only keys:")
    for k in sorted(missing)[:30]:
        print("  " + k)

    print("\nShape mismatches:")
    for k in shape_mismatches[:30]:
        print(
            f"  {k}: checkpoint={list(ckpt_sd[k].shape)} "
            f"model={list(model_sd[k].shape)}"
        )

    summary.update({
        "common_keys": len(common),
        "shape_matches": len(shape_matches),
        "shape_mismatches": len(shape_mismatches),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key", "status", "checkpoint_shape", "model_shape",
                "checkpoint_numel", "model_numel",
                "checkpoint_dtype", "model_dtype",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    exact = (
        len(ckpt_keys) == len(model_keys)
        and len(common) == len(ckpt_keys)
        and len(common) == len(model_keys)
        and len(shape_matches) == len(ckpt_keys)
        and len(shape_mismatches) == 0
        and len(missing) == 0
        and len(unexpected) == 0
    )

    print(f"\nEXACT KEY + SHAPE MATCH : {exact}")
    summary["exact_key_shape_match"] = exact

    print("\n5. STRICT CHECKPOINT LOAD")
    print("-" * 88)

    strict_load = False
    strict_error = None

    if exact:
        try:
            result = model.load_state_dict(ckpt_sd, strict=True)
            strict_load = True
            print("Strict load : PASS")
            print(f"Result      : {result}")
        except Exception as e:
            strict_error = f"{type(e).__name__}: {e}"
            print(f"Strict load : FAIL — {strict_error}")
    else:
        print("Strict load : NOT ATTEMPTED because exact key/shape match is false.")

    summary["strict_load"] = strict_load
    summary["strict_load_error"] = strict_error

    print("\n6. CPU FORWARD-PASS SMOKE TEST")
    print("-" * 88)

    cpu_forward = None
    if strict_load:
        try:
            x = torch.randn(1, 1, 64, 96, 96)
            with torch.no_grad():
                y = model(x)
            cpu_forward = forward_info(y)
            print("CPU forward : PASS")
            print(f"Output shape : {cpu_forward['shape']}")
            print(f"Channels     : {cpu_forward['channel_count']}")
            print(f"Logit min    : {cpu_forward['min']:.6f}")
            print(f"Logit max    : {cpu_forward['max']:.6f}")
            print(f"Logit mean   : {cpu_forward['mean']:.6f}")
            print(f"Logit std    : {cpu_forward['std']:.6f}")
            print(f"Pred classes : {cpu_forward['predicted_class_voxel_counts']}")
        except Exception as e:
            cpu_forward = {
                "status": f"FAIL: {type(e).__name__}: {e}"
            }
            print(f"CPU forward : FAIL — {type(e).__name__}: {e}")
    else:
        print("CPU forward : SKIPPED because strict load did not pass.")

    summary["cpu_forward"] = cpu_forward

    print("\n7. CUDA FORWARD-PASS SMOKE TEST")
    print("-" * 88)

    cuda_forward = None
    if strict_load and torch.cuda.is_available():
        try:
            cuda_model = model.cuda().eval()
            x = torch.randn(1, 1, 64, 96, 96, device="cuda")
            with torch.no_grad():
                y = cuda_model(x)
            cuda_forward = forward_info(y)
            print("CUDA forward : PASS")
            print(f"Output shape : {cuda_forward['shape']}")
            print(f"Channels     : {cuda_forward['channel_count']}")
            del x, y, cuda_model
            torch.cuda.empty_cache()
        except RuntimeError as e:
            cuda_forward = {
                "status": f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            }
            print(f"CUDA forward : RUNTIME ERROR — {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as e:
            cuda_forward = {
                "status": f"FAIL: {type(e).__name__}: {e}"
            }
            print(f"CUDA forward : FAIL — {type(e).__name__}: {e}")
    elif not strict_load:
        print("CUDA forward : SKIPPED because strict load did not pass.")
    else:
        print("CUDA forward : SKIPPED because CUDA is unavailable.")

    summary["cuda_forward"] = cuda_forward

    print("\n8. FINAL DIAGNOSIS")
    print("-" * 88)

    if exact and strict_load and cpu_forward and cpu_forward.get("channel_count") == 6:
        final_status = (
            "PASS — ORIGINAL_SWINUNETR_ARCHITECTURE_CONFIRMED_AND_CHECKPOINT_VALIDATED"
        )
        diagnosis = (
            "The selected Part 104 checkpoint exactly matches the historical "
            "MONAI SwinUNETR architecture (1 input channel, 6 output channels, "
            "feature_size 12). Strict loading succeeds and the model produces "
            "a 6-channel 3D segmentation output."
        )
    elif exact and not strict_load:
        final_status = "FAIL — EXACT_ARCHITECTURE_FOUND_BUT_STRICT_LOAD_FAILED"
        diagnosis = (
            "Key and shape structures match, but PyTorch strict loading failed."
        )
    elif exact and strict_load:
        final_status = "REVIEW_REQUIRED — FORWARD_PASS_FAILED"
        diagnosis = (
            "The checkpoint exactly matches and strict-loads into SwinUNETR, "
            "but the forward-pass test failed or produced an unexpected output."
        )
    else:
        final_status = "REVIEW_REQUIRED — CHECKPOINT_NOT_EXACTLY_COMPATIBLE"
        diagnosis = (
            "The direct historical SwinUNETR(1,6,12) architecture does not exactly "
            "match the checkpoint under the currently installed MONAI implementation."
        )

    print("Diagnosis:", diagnosis)
    print("Final status:", final_status)

    summary["diagnosis"] = diagnosis
    summary["final_status"] = final_status

    report = [
        "PART 109 — DIRECT Swin-UNETR CHECKPOINT LOAD & FORWARD VALIDATION",
        "=" * 88,
        f"Checkpoint: {CHECKPOINT}",
        f"SHA256: {sha}",
        f"MONAI version: {summary.get('monai_version')}",
        f"Model kwargs: {kwargs}",
        f"Checkpoint keys: {len(ckpt_sd)}",
        f"Checkpoint tensor elements: {tensor_numel(ckpt_sd):,}",
        f"Model keys: {len(model_sd)}",
        f"Model parameters: {parameter_count(model):,}",
        f"Exact key/shape match: {exact}",
        f"Strict load: {strict_load}",
        f"CPU forward: {cpu_forward}",
        f"CUDA forward: {cuda_forward}",
        "",
        "DIAGNOSIS",
        "-" * 88,
        diagnosis,
        "",
        "FINAL STATUS",
        "-" * 88,
        final_status,
        "",
        "No training, source modification, package installation, or checkpoint overwrite was performed.",
    ]
    TXT_PATH.write_text("\n".join(report), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    print("\n========================================================================================")
    print("PART 109 FINAL RESULT")
    print("========================================================================================")
    print(final_status)
    print("\nOUTPUTS:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(TXT_PATH)
    print("=" * 88)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
