"""
PART 110 — CANONICAL SwinUNETR SEGMENTATION INFERENCE FACTORY & SMOKE TEST

Purpose
-------
Establish the confirmed original MONAI SwinUNETR architecture as the canonical
inference path for the selected Part 104 checkpoint.

This part:
1. Loads the Part 104 final segmentation checkpoint read-only.
2. Constructs the exact architecture confirmed by Part 109.
3. Performs strict state-dict validation.
4. Runs deterministic CPU and CUDA inference smoke tests.
5. Verifies the 6-channel segmentation output.
6. Saves a reusable model specification and smoke-test summary.

IMPORTANT
---------
- No training is performed.
- No checkpoint is modified or overwritten.
- src/model.py is NOT modified.
- This script deliberately uses the confirmed MONAI SwinUNETR architecture.
- Smoke-test input is synthetic/deterministic; it is NOT a clinical inference result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part110_canonical_inference_factory"
)
REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

EXPECTED_SHA256 = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = REPORT_DIR / "part110_canonical_inference_summary.json"
REPORT_TXT = REPORT_DIR / "part110_canonical_inference_report.txt"
SPEC_JSON = OUTPUT_DIR / "part110_canonical_model_spec.json"
SMOKE_CSV = OUTPUT_DIR / "part110_smoke_test_results.csv"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_state_dict(checkpoint_obj: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Checkpoint object is not a dictionary.")

    candidates = [
        "model_state_dict",
        "state_dict",
        "model",
        "network_state_dict",
    ]

    for key in candidates:
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            tensor_values = [
                v for v in value.values() if isinstance(v, torch.Tensor)
            ]
            if tensor_values:
                return value, key

    tensor_values = [
        v for v in checkpoint_obj.values() if isinstance(v, torch.Tensor)
    ]
    if tensor_values:
        return checkpoint_obj, "<root>"

    raise RuntimeError("No tensor state dictionary found in checkpoint.")


def tensor_count(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(
        sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
    )


def build_canonical_model():
    """
    Exact architecture established by Part 109.

    The positional constructor is kept simple so this factory remains tied to
    the verified MONAI SwinUNETR implementation rather than src/model.py.
    """
    from monai.networks.nets import SwinUNETR

    try:
        model = SwinUNETR(
            in_channels=1,
            out_channels=6,
            feature_size=12,
            spatial_dims=3,
            use_checkpoint=False,
        )
    except TypeError:
        # Compatibility fallback for MONAI versions where spatial_dims is
        # implicit. Part 109 already confirmed the current environment.
        model = SwinUNETR(
            in_channels=1,
            out_channels=6,
            feature_size=12,
            use_checkpoint=False,
        )

    return model


def validate_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    model_state = model.state_dict()

    ckpt_keys = set(state_dict.keys())
    model_keys = set(model_state.keys())

    common = ckpt_keys & model_keys
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    shape_mismatches = []
    shape_matches = 0

    for key in sorted(common):
        ck_shape = tuple(state_dict[key].shape)
        model_shape = tuple(model_state[key].shape)
        if ck_shape == model_shape:
            shape_matches += 1
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(ck_shape),
                    "model_shape": list(model_shape),
                }
            )

    exact = (
        len(ckpt_keys) == len(model_keys)
        and len(common) == len(ckpt_keys)
        and len(common) == len(model_keys)
        and shape_matches == len(common)
        and not shape_mismatches
        and not missing
        and not unexpected
    )

    return {
        "checkpoint_keys": len(ckpt_keys),
        "model_keys": len(model_keys),
        "common_keys": len(common),
        "shape_matches": shape_matches,
        "shape_mismatches": shape_mismatches,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "exact_key_shape_match": bool(exact),
    }


def summarize_prediction(logits: torch.Tensor) -> Dict[str, Any]:
    pred = torch.argmax(logits, dim=1)
    unique, counts = torch.unique(pred, return_counts=True)

    distribution = {
        str(int(k)): int(v)
        for k, v in zip(unique.detach().cpu(), counts.detach().cpu())
    }

    return {
        "output_shape": list(logits.shape),
        "logit_min": float(logits.min().item()),
        "logit_max": float(logits.max().item()),
        "logit_mean": float(logits.mean().item()),
        "logit_std": float(logits.std().item()),
        "predicted_class_distribution": distribution,
    }


def run_forward(
    model,
    device: torch.device,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    model = model.to(device)
    model.eval()
    x = input_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return logits.detach(), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 88)
    print("PART 110 — CANONICAL SwinUNETR SEGMENTATION INFERENCE FACTORY & SMOKE TEST")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    results = []
    overall_pass = True

    # -----------------------------------------------------------------------
    # 1. Checkpoint
    # -----------------------------------------------------------------------
    print("\n1. CANONICAL CHECKPOINT")
    print("-" * 88)

    if not CHECKPOINT.exists():
        print("Checkpoint : NOT FOUND")
        print("FINAL STATUS: FAIL — CANONICAL_CHECKPOINT_MISSING")
        return 1

    actual_sha = sha256_file(CHECKPOINT)
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"SHA256     : {actual_sha}")
    print(f"Expected   : {EXPECTED_SHA256}")
    sha_pass = actual_sha == EXPECTED_SHA256
    print(f"SHA match  : {'PASS' if sha_pass else 'FAIL'}")

    if not sha_pass:
        overall_pass = False

    # -----------------------------------------------------------------------
    # 2. Load checkpoint
    # -----------------------------------------------------------------------
    print("\n2. READ-ONLY CHECKPOINT LOAD")
    print("-" * 88)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict, state_source = extract_state_dict(checkpoint)
    print(f"State source   : {state_source}")
    print(f"State keys     : {len(state_dict)}")
    print(f"Tensor elements: {tensor_count(state_dict):,}")

    # -----------------------------------------------------------------------
    # 3. MONAI + canonical factory
    # -----------------------------------------------------------------------
    print("\n3. CANONICAL MODEL FACTORY")
    print("-" * 88)

    try:
        import monai

        print(f"MONAI version : {monai.__version__}")
        model = build_canonical_model()

        param_count = sum(p.numel() for p in model.parameters())
        state_key_count = len(model.state_dict())
        state_tensor_count = tensor_count(model.state_dict())

        print("Architecture:")
        print("  MONAI SwinUNETR")
        print("  in_channels  = 1")
        print("  out_channels = 6")
        print("  feature_size = 12")
        print("  spatial_dims  = 3")
        print("  use_checkpoint = False")
        print(f"Parameters    : {param_count:,}")
        print(f"State keys    : {state_key_count}")
        print(f"State tensors : {state_tensor_count:,}")

        factory_pass = (
            param_count == 4_078_116
            and state_key_count == 159
            and state_tensor_count == 5_019_308
        )
        print(f"Factory spec  : {'PASS' if factory_pass else 'REVIEW'}")

        if not factory_pass:
            overall_pass = False

    except Exception as exc:
        print(f"Model factory : FAIL — {type(exc).__name__}: {exc}")
        overall_pass = False
        print("\nFINAL STATUS: FAIL — CANONICAL_MODEL_FACTORY_ERROR")
        return 1

    # -----------------------------------------------------------------------
    # 4. Exact architecture validation
    # -----------------------------------------------------------------------
    print("\n4. CHECKPOINT ↔ MODEL KEY/SHAPE VALIDATION")
    print("-" * 88)

    validation = validate_state_dict(model, state_dict)

    print(f"Checkpoint keys : {validation['checkpoint_keys']}")
    print(f"Model keys      : {validation['model_keys']}")
    print(f"Common keys     : {validation['common_keys']}")
    print(f"Shape matches   : {validation['shape_matches']}")
    print(f"Shape mismatches: {len(validation['shape_mismatches'])}")
    print(f"Missing keys    : {len(validation['missing_keys'])}")
    print(f"Unexpected keys : {len(validation['unexpected_keys'])}")
    print(
        "EXACT KEY + SHAPE MATCH : "
        f"{validation['exact_key_shape_match']}"
    )

    if not validation["exact_key_shape_match"]:
        overall_pass = False
        print("\nFINAL STATUS: FAIL — ARCHITECTURE_CHECKPOINT_MISMATCH")
        return 1

    # -----------------------------------------------------------------------
    # 5. Strict load
    # -----------------------------------------------------------------------
    print("\n5. STRICT CHECKPOINT LOAD")
    print("-" * 88)

    try:
        load_result = model.load_state_dict(state_dict, strict=True)
        print(f"Strict load : PASS")
        print(f"Result      : {load_result}")
        strict_pass = True
    except Exception as exc:
        print(f"Strict load : FAIL — {type(exc).__name__}: {exc}")
        strict_pass = False
        overall_pass = False

    if not strict_pass:
        print("\nFINAL STATUS: FAIL — STRICT_CHECKPOINT_LOAD_FAILED")
        return 1

    # -----------------------------------------------------------------------
    # 6. Deterministic synthetic volume
    # -----------------------------------------------------------------------
    print("\n6. DETERMINISTIC SMOKE-TEST INPUT")
    print("-" * 88)

    input_shape = (1, 1, 64, 96, 96)

    # A bounded deterministic tensor. This is intentionally NOT a patient image.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    smoke_input = torch.randn(input_shape, generator=generator, dtype=torch.float32)

    print(f"Input shape : {list(smoke_input.shape)}")
    print(f"Input dtype : {smoke_input.dtype}")
    print(f"Input min   : {smoke_input.min().item():.6f}")
    print(f"Input max   : {smoke_input.max().item():.6f}")
    print("Input type  : deterministic synthetic smoke-test volume")

    # -----------------------------------------------------------------------
    # 7. CPU inference
    # -----------------------------------------------------------------------
    print("\n7. CPU INFERENCE SMOKE TEST")
    print("-" * 88)

    try:
        cpu_logits, cpu_time = run_forward(
            model,
            torch.device("cpu"),
            smoke_input,
        )
        cpu_summary = summarize_prediction(cpu_logits)

        cpu_pass = cpu_logits.shape == (1, 6, 64, 96, 96)
        print("CPU forward : PASS" if cpu_pass else "CPU forward : FAIL")
        print(f"Output shape: {cpu_summary['output_shape']}")
        print(f"Time sec    : {cpu_time:.3f}")
        print(f"Logit min   : {cpu_summary['logit_min']:.6f}")
        print(f"Logit max   : {cpu_summary['logit_max']:.6f}")
        print(f"Logit mean  : {cpu_summary['logit_mean']:.6f}")
        print(f"Logit std   : {cpu_summary['logit_std']:.6f}")
        print(
            "Pred classes: "
            + json.dumps(
                cpu_summary["predicted_class_distribution"],
                sort_keys=True,
            )
        )

        if not cpu_pass:
            overall_pass = False

    except Exception as exc:
        print(f"CPU forward : FAIL — {type(exc).__name__}: {exc}")
        cpu_summary = {"error": str(exc)}
        cpu_time = None
        overall_pass = False

    # -----------------------------------------------------------------------
    # 8. CUDA inference
    # -----------------------------------------------------------------------
    print("\n8. CUDA INFERENCE SMOKE TEST")
    print("-" * 88)

    cuda_summary = None
    cuda_time = None

    if torch.cuda.is_available():
        try:
            cuda_model = build_canonical_model()
            cuda_model.load_state_dict(state_dict, strict=True)

            cuda_logits, cuda_time = run_forward(
                cuda_model,
                torch.device("cuda:0"),
                smoke_input,
            )
            cuda_summary = summarize_prediction(cuda_logits)

            cuda_pass = cuda_logits.shape == (1, 6, 64, 96, 96)

            print("CUDA forward : PASS" if cuda_pass else "CUDA forward : FAIL")
            print(f"Output shape : {cuda_summary['output_shape']}")
            print(f"Time sec     : {cuda_time:.3f}")
            print(
                "Pred classes : "
                + json.dumps(
                    cuda_summary["predicted_class_distribution"],
                    sort_keys=True,
                )
            )

            if not cuda_pass:
                overall_pass = False

            del cuda_model, cuda_logits
            torch.cuda.empty_cache()

        except Exception as exc:
            print(f"CUDA forward : FAIL — {type(exc).__name__}: {exc}")
            cuda_summary = {"error": str(exc)}
            overall_pass = False
    else:
        print("CUDA forward : SKIPPED — CUDA unavailable")
        cuda_summary = {"skipped": True, "reason": "CUDA unavailable"}

    # -----------------------------------------------------------------------
    # 9. Save canonical specification
    # -----------------------------------------------------------------------
    print("\n9. SAVE CANONICAL INFERENCE SPECIFICATION")
    print("-" * 88)

    spec = {
        "part": 110,
        "status": "PASS" if overall_pass else "FAIL",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_sha,
        "architecture": {
            "name": "MONAI SwinUNETR",
            "in_channels": 1,
            "out_channels": 6,
            "feature_size": 12,
            "spatial_dims": 3,
            "use_checkpoint": False,
            "full_volume_shape": [64, 96, 96],
            "training_crop_shape": [32, 64, 64],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "validation": validation,
        "smoke_test": {
            "seed": SEED,
            "input_shape": list(input_shape),
            "cpu": cpu_summary,
            "cpu_time_sec": cpu_time,
            "cuda": cuda_summary,
            "cuda_time_sec": cuda_time,
        },
        "scientific_note": (
            "This part validates the inference architecture and runtime "
            "using a deterministic synthetic input. It is not a clinical "
            "performance evaluation and does not establish expert-ground-"
            "truth segmentation accuracy."
        ),
    }

    SPEC_JSON.write_text(
        json.dumps(spec, indent=2),
        encoding="utf-8",
    )

    smoke_rows = [
        {
            "device": "cpu",
            "status": "PASS" if "error" not in cpu_summary else "FAIL",
            "input_shape": str(list(input_shape)),
            "output_shape": str(cpu_summary.get("output_shape")),
            "time_sec": cpu_time,
            "logit_min": cpu_summary.get("logit_min"),
            "logit_max": cpu_summary.get("logit_max"),
            "logit_mean": cpu_summary.get("logit_mean"),
            "logit_std": cpu_summary.get("logit_std"),
            "predicted_class_distribution": json.dumps(
                cpu_summary.get("predicted_class_distribution", {}),
                sort_keys=True,
            ),
        }
    ]

    if torch.cuda.is_available():
        smoke_rows.append(
            {
                "device": "cuda:0",
                "status": (
                    "PASS"
                    if cuda_summary and "error" not in cuda_summary
                    else "FAIL"
                ),
                "input_shape": str(list(input_shape)),
                "output_shape": str(
                    cuda_summary.get("output_shape")
                    if cuda_summary
                    else None
                ),
                "time_sec": cuda_time,
                "logit_min": (
                    cuda_summary.get("logit_min")
                    if cuda_summary and "error" not in cuda_summary
                    else None
                ),
                "logit_max": (
                    cuda_summary.get("logit_max")
                    if cuda_summary and "error" not in cuda_summary
                    else None
                ),
                "logit_mean": (
                    cuda_summary.get("logit_mean")
                    if cuda_summary and "error" not in cuda_summary
                    else None
                ),
                "logit_std": (
                    cuda_summary.get("logit_std")
                    if cuda_summary and "error" not in cuda_summary
                    else None
                ),
                "predicted_class_distribution": json.dumps(
                    cuda_summary.get("predicted_class_distribution", {})
                    if cuda_summary
                    else {},
                    sort_keys=True,
                ),
            }
        )

    with SMOKE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=smoke_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(smoke_rows)

    # -----------------------------------------------------------------------
    # 10. Final report
    # -----------------------------------------------------------------------
    status_line = (
        "PASS — CANONICAL_SWINUNETR_INFERENCE_PATH_ESTABLISHED"
        if overall_pass
        else "FAIL — CANONICAL_SWINUNETR_INFERENCE_PATH_REQUIRES_REVIEW"
    )

    report = f"""
PART 110 — CANONICAL SwinUNETR SEGMENTATION INFERENCE FACTORY & SMOKE TEST
=========================================================================

Final status:
{status_line}

Checkpoint:
{CHECKPOINT}
SHA256:
{actual_sha}

Confirmed architecture:
- MONAI SwinUNETR
- in_channels = 1
- out_channels = 6
- feature_size = 12
- spatial_dims = 3
- use_checkpoint = False
- full volume = (64, 96, 96)
- training crop = (32, 64, 64)

Checkpoint/model validation:
- checkpoint keys = {validation['checkpoint_keys']}
- model keys = {validation['model_keys']}
- common keys = {validation['common_keys']}
- shape matches = {validation['shape_matches']}
- shape mismatches = {len(validation['shape_mismatches'])}
- missing keys = {len(validation['missing_keys'])}
- unexpected keys = {len(validation['unexpected_keys'])}
- exact key + shape match = {validation['exact_key_shape_match']}

Strict checkpoint load:
PASS

CPU smoke test:
{json.dumps(cpu_summary, indent=2)}
CPU time:
{cpu_time}

CUDA smoke test:
{json.dumps(cuda_summary, indent=2)}
CUDA time:
{cuda_time}

IMPORTANT SCIENTIFIC NOTE
-------------------------
The smoke test uses a deterministic synthetic 3D tensor solely to validate
model construction, checkpoint loading, tensor dimensions, and runtime
inference. It is not a patient prediction and must not be interpreted as
clinical performance or segmentation accuracy.

Part 110 does not train the model, modify the checkpoint, or modify src/model.py.
"""

    REPORT_TXT.write_text(report.strip() + "\n", encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PART 110 FINAL RESULT")
    print("=" * 88)
    print(status_line)
    print("\nOUTPUTS:")
    print(SPEC_JSON)
    print(SMOKE_CSV)
    print(REPORT_TXT)
    print(SUMMARY_JSON)
    print("=" * 88)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
