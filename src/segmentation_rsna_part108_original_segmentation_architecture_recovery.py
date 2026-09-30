"""
PART 108 — ORIGINAL SEGMENTATION ARCHITECTURE RECOVERY & COMPATIBILITY TEST

Purpose
-------
Recover the architecture represented by the selected Part 104 checkpoint without
modifying the project or checkpoint.

The Part 107.1 audit established that the checkpoint is a 3D Swin-UNETR-style
segmentation model (swinViT + encoders + decoders + out), while the current
src.model.create_model() builds a different 2D/feature-classifier architecture.

This script therefore:
1. Inspects the selected checkpoint metadata and state-dict.
2. Searches project Python source for the original segmentation architecture
   implementation and training/model construction clues.
3. Checks installed MONAI availability/version.
4. Tries to identify a compatible SwinUNETR constructor/configuration from the
   recovered project evidence.
5. Constructs candidate segmentation models only in memory.
6. Compares candidate state-dict keys/shapes with the checkpoint.
7. Attempts strict loading only for an exact key/shape match.
8. Runs a small forward-pass smoke test only after an exact strict load.
9. Writes detailed reports.

SAFETY
------
- Does NOT train.
- Does NOT edit src/model.py.
- Does NOT edit/delete/overwrite any checkpoint.
- Does NOT install packages.
- Does NOT replace the current model factory.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import inspect
import json
import re
import sys
import traceback
from collections import Counter
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
    / "rsna_part108_original_segmentation_architecture_recovery"
)
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "part108_candidate_compatibility.csv"
JSON_PATH = REPORT_DIR / "part108_architecture_recovery_summary.json"
TXT_PATH = REPORT_DIR / "part108_architecture_recovery_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "numel": int(x.numel()),
        }
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(obj: Any):
    candidates = []
    if isinstance(obj, dict):
        candidates.append(("root", obj))
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
            "model_weights",
        ):
            value = obj.get(key)
            if isinstance(value, dict):
                candidates.append((key, value))

    best_name, best = None, None
    score = -1
    for name, candidate in candidates:
        items = [
            (k, v)
            for k, v in candidate.items()
            if isinstance(k, str) and torch.is_tensor(v)
        ]
        if len(items) > score:
            score = len(items)
            best_name = name
            best = {k: v for k, v in items}

    if best is None:
        raise RuntimeError("No tensor state_dict found.")
    return best_name, best


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def tensor_numel(sd: dict) -> int:
    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))


def inspect_python_files():
    results = []
    keywords = (
        "SwinUNETR",
        "SwinUNETR",
        "swinViT",
        "swin_unetr",
        "SwinUNETR",
        "feature_size",
        "num_classes",
        "out_channels",
        "img_size",
        "full_shape",
        "crop_shape",
        "MONAI",
        "monai",
        "Swin",
        "decoder",
        "encoder",
    )

    for path in sorted(SRC_DIR.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            results.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue

        hits = {kw: len(re.findall(re.escape(kw), text, flags=re.IGNORECASE)) for kw in keywords}
        hits = {k: v for k, v in hits.items() if v}
        if hits:
            snippets = []
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if any(k.lower() in line.lower() for k in keywords):
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    snippets.append(
                        {
                            "line": i + 1,
                            "text": "\n".join(lines[start:end]),
                        }
                    )
                    if len(snippets) >= 12:
                        break

            results.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "hits": hits,
                    "snippets": snippets,
                }
            )

    return results


def import_maybe(name):
    try:
        module = importlib.import_module(name)
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
            "file": getattr(module, "__file__", None),
            "module": module,
            "error": None,
        }
    except Exception as e:
        return {
            "available": False,
            "version": None,
            "file": None,
            "module": None,
            "error": f"{type(e).__name__}: {e}",
        }


def candidate_constructor_candidates():
    candidates = []

    # MONAI is the canonical candidate if installed.
    monai_info = import_maybe("monai")
    if monai_info["available"]:
        try:
            from monai.networks.nets import SwinUNETR
            candidates.append(
                {
                    "name": "monai.networks.nets.SwinUNETR",
                    "constructor": SwinUNETR,
                    "source": "MONAI",
                    "signature": str(inspect.signature(SwinUNETR)),
                }
            )
        except Exception as e:
            candidates.append(
                {
                    "name": "monai.networks.nets.SwinUNETR",
                    "constructor": None,
                    "source": "MONAI",
                    "signature": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    # Search src modules for classes with Swin/UNETR/segmentation names.
    for py in sorted(SRC_DIR.glob("*.py")):
        module_name = f"src.{py.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                low = name.lower()
                if any(token in low for token in ("swin", "unetr", "segment")):
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = None
                    candidates.append(
                        {
                            "name": f"{module_name}.{name}",
                            "constructor": obj,
                            "source": "project",
                            "signature": sig,
                        }
                    )

    # De-duplicate by fully-qualified name.
    unique = {}
    for c in candidates:
        unique[c["name"]] = c
    return list(unique.values()), monai_info


def construct_candidate(constructor, metadata):
    """
    Try conservative configurations. These are common MONAI SwinUNETR-style
    configurations, but no candidate is accepted unless its complete state_dict
    matches the checkpoint.
    """
    attempts = []

    num_classes = int(metadata.get("num_classes", 6))
    feature_size = int(metadata.get("feature_size", 12))
    full_shape = tuple(metadata.get("full_shape", [64, 96, 96]))
    crop_shape = tuple(metadata.get("crop_shape", [32, 64, 64]))

    # Modern MONAI accepts img_size=None in many versions; older releases may
    # require/remove img_size. Try metadata-derived and minimal forms.
    kwargs_list = [
        {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": feature_size,
            "spatial_dims": 3,
            "use_checkpoint": False,
        },
        {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": feature_size,
            "use_checkpoint": False,
        },
        {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": feature_size,
        },
        {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": feature_size,
            "img_size": full_shape,
            "use_checkpoint": False,
        },
        {
            "in_channels": 1,
            "out_channels": num_classes,
            "feature_size": feature_size,
            "img_size": crop_shape,
            "use_checkpoint": False,
        },
    ]

    # Inspect signature and filter unsupported kwargs.
    try:
        sig = inspect.signature(constructor)
        supported = set(sig.parameters)
    except Exception:
        supported = set()

    for kwargs in kwargs_list:
        if supported:
            filtered = {k: v for k, v in kwargs.items() if k in supported}
        else:
            filtered = kwargs

        label = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
        try:
            model = constructor(**filtered)
            if isinstance(model, torch.nn.Module):
                attempts.append(
                    {
                        "label": label,
                        "success": True,
                        "model": model,
                        "error": None,
                    }
                )
            else:
                attempts.append(
                    {
                        "label": label,
                        "success": False,
                        "model": None,
                        "error": f"returned {type(model).__name__}",
                    }
                )
        except Exception as e:
            attempts.append(
                {
                    "label": label,
                    "success": False,
                    "model": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return attempts


def compare_state_dict(ckpt_sd, model_sd):
    ck = set(ckpt_sd)
    mk = set(model_sd)
    common = ck & mk
    missing = mk - ck
    unexpected = ck - mk

    matches = []
    mismatches = []

    for key in sorted(common):
        cs = tuple(ckpt_sd[key].shape)
        ms = tuple(model_sd[key].shape)
        if cs == ms:
            matches.append(key)
        else:
            mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(cs),
                    "model_shape": list(ms),
                }
            )

    return {
        "common": len(common),
        "matches": len(matches),
        "shape_mismatches": len(mismatches),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": mismatches,
    }


def run_forward(model):
    model = model.cpu().eval()
    # The Part 98 checkpoint metadata records full_shape [64,96,96].
    x = torch.randn(1, 1, 64, 96, 96)
    with torch.no_grad():
        y = model(x)

    if isinstance(y, (tuple, list)):
        primary = y[0]
        output_type = type(y).__name__
    elif isinstance(y, dict):
        # Try common segmentation output names.
        primary = None
        for key in ("out", "logits", "pred", "seg"):
            if key in y and torch.is_tensor(y[key]):
                primary = y[key]
                break
        if primary is None:
            raise RuntimeError("Model returned dict without a recognized tensor output.")
        output_type = "dict"
    else:
        primary = y
        output_type = type(y).__name__

    if not torch.is_tensor(primary):
        raise RuntimeError(f"Primary output is not a tensor: {type(primary).__name__}")

    return {
        "output_type": output_type,
        "output_shape": list(primary.shape),
        "output_dtype": str(primary.dtype),
        "expected_channels": 6,
        "channel_match": primary.ndim >= 2 and int(primary.shape[1]) == 6,
    }


def main():
    print("=" * 88)
    print("PART 108 — ORIGINAL SEGMENTATION ARCHITECTURE RECOVERY & COMPATIBILITY")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print()

    summary = {
        "part": 108,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": str(CHECKPOINT),
    }

    print("1. SELECTED CHECKPOINT")
    print("-" * 88)
    if not CHECKPOINT.exists():
        print("Checkpoint : MISSING")
        print("FINAL: FAIL — CHECKPOINT_MISSING")
        return 1

    sha = sha256_file(CHECKPOINT)
    print("Checkpoint : FOUND")
    print(f"SHA256     : {sha}")

    ckpt = load_checkpoint(CHECKPOINT)
    source, ckpt_sd = extract_state_dict(ckpt)

    metadata = {}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if k not in {
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "network",
                "weights",
                "model_weights",
                "optimizer_state_dict",
                "scaler_state_dict",
            } and not torch.is_tensor(v):
                metadata[k] = safe_json(v)

    print(f"State source       : {source}")
    print(f"Tensor keys        : {len(ckpt_sd)}")
    print(f"Tensor elements    : {tensor_numel(ckpt_sd):,}")

    for key in ("feature_size", "num_classes", "full_shape", "crop_shape", "epoch"):
        if key in metadata:
            print(f"{key:<18}: {metadata[key]}")

    summary["checkpoint_sha256"] = sha
    summary["checkpoint_state_source"] = source
    summary["checkpoint_key_count"] = len(ckpt_sd)
    summary["checkpoint_tensor_numel"] = tensor_numel(ckpt_sd)
    summary["checkpoint_metadata"] = metadata

    print("\n2. PROJECT SOURCE RECOVERY SCAN")
    print("-" * 88)
    source_scan = inspect_python_files()
    summary["project_source_scan"] = source_scan

    if source_scan:
        for item in source_scan:
            print(f"\n{item['file']}")
            if "error" in item:
                print("  ERROR:", item["error"])
                continue
            print("  hits:", item["hits"])
            for snippet in item["snippets"][:4]:
                print(f"  line {snippet['line']}:")
                for line in snippet["text"].splitlines():
                    print("    " + line)
    else:
        print("No matching segmentation/Swin/UNETR source clues found in src/*.py.")

    print("\n3. INSTALLED SEGMENTATION FRAMEWORKS")
    print("-" * 88)
    monai_info = import_maybe("monai")
    print(f"MONAI available : {monai_info['available']}")
    print(f"MONAI version   : {monai_info['version']}")
    print(f"MONAI location  : {monai_info['file']}")
    if monai_info["error"]:
        print(f"MONAI error     : {monai_info['error']}")

    summary["monai"] = {
        k: v for k, v in monai_info.items() if k != "module"
    }

    print("\n4. SEGMENTATION CONSTRUCTOR DISCOVERY")
    print("-" * 88)
    candidates, _ = candidate_constructor_candidates()

    if not candidates:
        print("No candidate Swin/UNETR segmentation constructor discovered.")
        summary["candidate_count"] = 0
        summary["final_status"] = "REVIEW_REQUIRED — NO_SEGMENTATION_CONSTRUCTOR_FOUND"
        JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(
            "PART 108\n\nNO SEGMENTATION CONSTRUCTOR FOUND.\n",
            encoding="utf-8",
        )
        print("\nFINAL: REVIEW_REQUIRED — NO_SEGMENTATION_CONSTRUCTOR_FOUND")
        return 0

    for c in candidates:
        print(f"Candidate : {c['name']}")
        print(f"Source    : {c.get('source')}")
        print(f"Signature : {c.get('signature')}")
        if c.get("error"):
            print(f"Error     : {c['error']}")
    summary["candidate_count"] = len(candidates)

    print("\n5. CANDIDATE ARCHITECTURE CONSTRUCTION & CHECKPOINT MATCH")
    print("-" * 88)

    candidate_rows = []
    exact_candidates = []

    for candidate in candidates:
        constructor = candidate.get("constructor")
        if constructor is None:
            continue

        attempts = construct_candidate(constructor, metadata)

        for attempt_index, attempt in enumerate(attempts, start=1):
            row = {
                "candidate": candidate["name"],
                "source": candidate.get("source"),
                "signature": candidate.get("signature"),
                "attempt": attempt_index,
                "configuration": attempt["label"],
                "constructed": attempt["success"],
                "model_class": "",
                "model_parameters": "",
                "model_state_keys": "",
                "common_keys": "",
                "shape_matches": "",
                "shape_mismatches": "",
                "missing_keys": "",
                "unexpected_keys": "",
                "strict_load": "NOT_ATTEMPTED",
                "forward_pass": "NOT_ATTEMPTED",
                "forward_shape": "",
                "error": attempt.get("error") or "",
            }

            if not attempt["success"]:
                candidate_rows.append(row)
                print(f"\n[{candidate['name']}]")
                print(f"  Attempt {attempt_index}: construction FAIL")
                print(f"  {attempt.get('error')}")
                continue

            model = attempt["model"]
            model_sd = model.state_dict()
            comparison = compare_state_dict(ckpt_sd, model_sd)

            row["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
            row["model_parameters"] = count_parameters(model)
            row["model_state_keys"] = len(model_sd)
            row["common_keys"] = comparison["common"]
            row["shape_matches"] = comparison["matches"]
            row["shape_mismatches"] = comparison["shape_mismatches"]
            row["missing_keys"] = comparison["missing"]
            row["unexpected_keys"] = comparison["unexpected"]

            exact = (
                comparison["common"] == len(ckpt_sd)
                and comparison["common"] == len(model_sd)
                and comparison["matches"] == len(ckpt_sd)
                and comparison["shape_mismatches"] == 0
                and comparison["missing"] == 0
                and comparison["unexpected"] == 0
            )

            print(f"\n[{candidate['name']}]")
            print(f"  Attempt {attempt_index}: {attempt['label']}")
            print(f"  Model class       : {row['model_class']}")
            print(f"  Model parameters  : {row['model_parameters']:,}")
            print(f"  Model state keys  : {row['model_state_keys']}")
            print(f"  Common keys       : {row['common_keys']}")
            print(f"  Shape matches     : {row['shape_matches']}")
            print(f"  Shape mismatches  : {row['shape_mismatches']}")
            print(f"  Missing keys      : {row['missing_keys']}")
            print(f"  Unexpected keys   : {row['unexpected_keys']}")
            print(f"  EXACT MATCH       : {exact}")

            if exact:
                try:
                    model.load_state_dict(ckpt_sd, strict=True)
                    row["strict_load"] = "PASS"
                    exact_candidates.append(
                        {
                            "candidate": candidate,
                            "attempt": attempt,
                            "model": model,
                            "comparison": comparison,
                        }
                    )
                    print("  Strict load       : PASS")
                except Exception as e:
                    row["strict_load"] = f"FAIL: {type(e).__name__}: {e}"
                    print(f"  Strict load       : FAIL — {type(e).__name__}: {e}")
            else:
                # Never force-load an incompatible checkpoint.
                row["strict_load"] = "NOT_ATTEMPTED_DUE_TO_MISMATCH"

            candidate_rows.append(row)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "source",
                "signature",
                "attempt",
                "configuration",
                "constructed",
                "model_class",
                "model_parameters",
                "model_state_keys",
                "common_keys",
                "shape_matches",
                "shape_mismatches",
                "missing_keys",
                "unexpected_keys",
                "strict_load",
                "forward_pass",
                "forward_shape",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary["candidate_results"] = candidate_rows

    print("\n6. EXACT MATCH FORWARD-PASS TEST")
    print("-" * 88)

    forward_results = []
    if not exact_candidates:
        print("No exact checkpoint-compatible segmentation architecture was found.")
        summary["exact_match_found"] = False
    else:
        summary["exact_match_found"] = True

        for item in exact_candidates:
            model = item["model"]
            try:
                result = run_forward(model)
                print(f"Candidate: {item['candidate']['name']}")
                print(f"Forward pass : PASS")
                print(f"Output shape : {result['output_shape']}")
                print(f"Channels     : {result['channel_match']}")
                forward_results.append(
                    {
                        "candidate": item["candidate"]["name"],
                        "status": "PASS",
                        **result,
                    }
                )
            except Exception as e:
                print(f"Candidate: {item['candidate']['name']}")
                print(f"Forward pass : FAIL — {type(e).__name__}: {e}")
                forward_results.append(
                    {
                        "candidate": item["candidate"]["name"],
                        "status": f"FAIL: {type(e).__name__}: {e}",
                    }
                )

    summary["forward_results"] = forward_results

    print("\n7. FINAL DIAGNOSIS")
    print("-" * 88)

    if exact_candidates and any(r.get("status") == "PASS" and r.get("channel_match") for r in forward_results):
        final_status = "PASS — ORIGINAL_SEGMENTATION_ARCHITECTURE_RECOVERED_AND_CHECKPOINT_LOADS"
        diagnosis = (
            "A segmentation architecture was reconstructed that exactly matches "
            "the selected Part 104 checkpoint and successfully produces a 6-channel "
            "3D segmentation output."
        )
    elif exact_candidates:
        final_status = "REVIEW_REQUIRED — CHECKPOINT_MATCHES_BUT_FORWARD_TEST_FAILED"
        diagnosis = (
            "An exact checkpoint-compatible architecture was found and strict-loaded, "
            "but the forward-pass smoke test did not complete successfully."
        )
    else:
        final_status = "REVIEW_REQUIRED — ORIGINAL_ARCHITECTURE_NOT_YET_RECONSTRUCTED"
        diagnosis = (
            "No tested candidate exactly matches the checkpoint. The checkpoint remains "
            "untouched; additional recovery from the Part 98 training/model source is required."
        )

    print("Diagnosis:", diagnosis)
    print("Final status:", final_status)

    summary["diagnosis"] = diagnosis
    summary["final_status"] = final_status

    report_lines = [
        "PART 108 — ORIGINAL SEGMENTATION ARCHITECTURE RECOVERY",
        "=" * 88,
        f"Project root: {PROJECT_ROOT}",
        f"Checkpoint: {CHECKPOINT}",
        f"Checkpoint SHA256: {sha}",
        f"Checkpoint keys: {len(ckpt_sd)}",
        f"Checkpoint tensor elements: {tensor_numel(ckpt_sd):,}",
        f"MONAI available: {monai_info['available']}",
        f"MONAI version: {monai_info['version']}",
        f"Candidate constructors: {len(candidates)}",
        f"Exact architecture match: {summary['exact_match_found']}",
        "",
        "CHECKPOINT METADATA",
        "-" * 88,
    ]

    for key in ("feature_size", "num_classes", "full_shape", "crop_shape", "epoch"):
        if key in metadata:
            report_lines.append(f"{key}: {metadata[key]}")

    report_lines += [
        "",
        "FINAL DIAGNOSIS",
        "-" * 88,
        diagnosis,
        "",
        "FINAL STATUS",
        "-" * 88,
        final_status,
        "",
        "This diagnostic did not train, modify source files, install packages, or overwrite the checkpoint.",
    ]

    TXT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print("\n========================================================================================")
    print("PART 108 FINAL RESULT")
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
