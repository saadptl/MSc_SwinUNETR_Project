"""
PART 107.1 — CORRECTED MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY DIAGNOSIS

Purpose:
    Corrects Part 107's project-root import-path issue and performs the actual
    compatibility diagnosis between the Part 104 final checkpoint and the
    current src.model.create_model() factory.

AUDIT ONLY:
    - No training
    - No checkpoint modification
    - No source-code modification
    - No package installation
    - No overwrite of the selected final checkpoint
"""

from __future__ import annotations

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part107_1_corrected_model_compatibility_diagnosis"
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "part107_1_key_compatibility.csv"
JSON_PATH = REPORT_DIR / "part107_1_model_compatibility_summary.json"
TXT_PATH = REPORT_DIR / "part107_1_model_compatibility_report.txt"


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

    best = None
    best_name = None
    best_score = -1
    for name, candidate in candidates:
        tensor_items = [
            (k, v)
            for k, v in candidate.items()
            if isinstance(k, str) and torch.is_tensor(v)
        ]
        if len(tensor_items) > best_score:
            best_score = len(tensor_items)
            best_name = name
            best = {k: v for k, v in tensor_items}

    if best is None:
        raise RuntimeError("No tensor state_dict-like object found.")
    return best_name, best


def safe_json(x):
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


def strip_prefix(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "net.",
        "network.",
        "_orig_mod.",
        "state_dict.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def model_param_count(model):
    return sum(p.numel() for p in model.parameters())


def tensor_numel(sd):
    return sum(v.numel() for v in sd.values() if torch.is_tensor(v))


def find_factory():
    errors = []
    try:
        module = importlib.import_module("src.model")
    except Exception as e:
        return None, None, [f"src.model import failed: {type(e).__name__}: {e}"]

    for name in ("create_model", "build_model", "get_model", "make_model"):
        fn = getattr(module, name, None)
        if callable(fn):
            return module, fn, errors

    candidates = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and any(s in name.lower() for s in ("model", "create", "build")):
            candidates.append(name)

    errors.append("No standard model factory found: " + ", ".join(candidates))
    return module, None, errors


def construct_model(factory):
    attempts = [
        ("factory()", lambda: factory()),
        ("factory(num_classes=6)", lambda: factory(num_classes=6)),
        ("factory(in_channels=1, out_channels=6)", lambda: factory(in_channels=1, out_channels=6)),
        (
            "factory(in_channels=1, out_channels=6, feature_size=12)",
            lambda: factory(in_channels=1, out_channels=6, feature_size=12),
        ),
        (
            "factory(in_channels=1, out_channels=6, img_size=(64,96,96), feature_size=12)",
            lambda: factory(
                in_channels=1,
                out_channels=6,
                img_size=(64, 96, 96),
                feature_size=12,
            ),
        ),
    ]
    errors = []
    for label, fn in attempts:
        try:
            m = fn()
            if isinstance(m, torch.nn.Module):
                return m, label, errors
            errors.append(f"{label}: returned {type(m).__name__}")
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    return None, None, errors


def main():
    print("=" * 88)
    print("PART 107.1 — CORRECTED MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print()

    summary = {
        "part": "107.1",
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "checkpoint": str(CHECKPOINT),
    }

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------
    print("1. FINAL CHECKPOINT")
    print("-" * 88)
    if not CHECKPOINT.exists():
        print("Checkpoint : MISSING")
        print("FINAL: FAIL — FINAL_CHECKPOINT_MISSING")
        return 1

    sha = sha256_file(CHECKPOINT)
    size_mb = CHECKPOINT.stat().st_size / (1024 * 1024)
    print("Checkpoint : FOUND")
    print(f"Size MB    : {size_mb:.3f}")
    print(f"SHA256     : {sha}")

    summary["checkpoint_sha256"] = sha
    summary["checkpoint_size_mb"] = size_mb

    ckpt = load_checkpoint(CHECKPOINT)
    source, ckpt_sd = extract_state_dict(ckpt)

    print("\n2. CHECKPOINT METADATA")
    print("-" * 88)
    print(f"State-dict source      : {source}")
    print(f"Checkpoint tensor keys : {len(ckpt_sd)}")
    print(f"Tensor elements        : {tensor_numel(ckpt_sd):,}")

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
            } and not torch.is_tensor(v):
                metadata[k] = safe_json(v)

    for k in sorted(metadata):
        value = json.dumps(metadata[k], ensure_ascii=False)
        if len(value) > 500:
            value = value[:500] + "..."
        print(f"  {k}: {value}")

    summary["checkpoint_state_dict_source"] = source
    summary["checkpoint_key_count"] = len(ckpt_sd)
    summary["checkpoint_tensor_numel"] = tensor_numel(ckpt_sd)
    summary["checkpoint_metadata"] = metadata

    # -------------------------------------------------------------------------
    # Import current factory
    # -------------------------------------------------------------------------
    print("\n3. CURRENT PROJECT MODEL FACTORY")
    print("-" * 88)
    module, factory, errors = find_factory()
    if module is None or factory is None:
        print("Factory : FAIL")
        for e in errors:
            print("  " + e)
        print("\nFINAL: FAIL — MODEL_FACTORY_UNAVAILABLE")
        return 1

    print(f"Factory module : {inspect.getsourcefile(module)}")
    print(f"Factory        : {factory.__name__}")
    try:
        print(f"Signature      : {inspect.signature(factory)}")
    except Exception:
        print("Signature      : unavailable")

    summary["factory_module"] = inspect.getsourcefile(module)
    summary["factory_name"] = factory.__name__
    try:
        summary["factory_signature"] = str(inspect.signature(factory))
    except Exception:
        summary["factory_signature"] = None

    try:
        source_text = inspect.getsource(module)
        summary["model_source_keywords"] = {
            x: (x.lower() in source_text.lower())
            for x in (
                "SwinUNETR",
                "swin_unetr",
                "UNETR",
                "timm",
                "feature_size",
                "num_classes",
                "out_channels",
                "in_channels",
            )
        }
        print("Source keywords:")
        for k, v in summary["model_source_keywords"].items():
            print(f"  {k}: {v}")
    except Exception as e:
        summary["model_source_error"] = str(e)

    # -------------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------------
    print("\n4. CURRENT MODEL CONSTRUCTION")
    print("-" * 88)
    model, call, construct_errors = construct_model(factory)
    if model is None:
        print("Model construction : FAIL")
        for e in construct_errors:
            print("  " + e)
        print("\nFINAL: FAIL — MODEL_CONSTRUCTION_FAILED")
        return 1

    model_sd = model.state_dict()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = model_param_count(model)

    print(f"Construction call     : {call}")
    print(f"Model class           : {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"Trainable parameters  : {trainable:,}")
    print(f"All model parameters  : {all_params:,}")
    print(f"Model state keys      : {len(model_sd)}")
    print(f"Model tensor elements : {tensor_numel(model_sd):,}")

    summary["factory_call"] = call
    summary["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
    summary["model_trainable_parameters"] = trainable
    summary["model_all_parameters"] = all_params
    summary["model_state_key_count"] = len(model_sd)
    summary["model_state_tensor_numel"] = tensor_numel(model_sd)

    # -------------------------------------------------------------------------
    # Exact compatibility
    # -------------------------------------------------------------------------
    print("\n5. EXACT CHECKPOINT ↔ MODEL COMPARISON")
    print("-" * 88)

    ckpt_keys = set(ckpt_sd)
    model_keys = set(model_sd)

    common = sorted(ckpt_keys & model_keys)
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    shape_matches = []
    shape_mismatches = []

    for k in common:
        a = tuple(ckpt_sd[k].shape)
        b = tuple(model_sd[k].shape)
        if a == b:
            shape_matches.append(k)
        else:
            shape_mismatches.append(
                {
                    "key": k,
                    "checkpoint_shape": list(a),
                    "model_shape": list(b),
                    "checkpoint_numel": int(ckpt_sd[k].numel()),
                    "model_numel": int(model_sd[k].numel()),
                }
            )

    print(f"Common keys             : {len(common)}")
    print(f"Shape-compatible keys   : {len(shape_matches)}")
    print(f"Shape mismatches        : {len(shape_mismatches)}")
    print(f"Missing model keys      : {len(missing)}")
    print(f"Unexpected checkpoint   : {len(unexpected)}")

    summary.update(
        {
            "common_keys": len(common),
            "shape_compatible_keys": len(shape_matches),
            "shape_mismatch_count": len(shape_mismatches),
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
        }
    )

    print("\nFirst 40 MISSING model keys:")
    for k in missing[:40]:
        print("  " + k)

    print("\nFirst 40 UNEXPECTED checkpoint keys:")
    for k in unexpected[:40]:
        print("  " + k)

    print("\nSHAPE MISMATCHES:")
    for x in shape_mismatches[:40]:
        print(
            f"  {x['key']}: checkpoint={x['checkpoint_shape']} "
            f"model={x['model_shape']}"
        )

    # -------------------------------------------------------------------------
    # Normalized/prefix compatibility
    # -------------------------------------------------------------------------
    print("\n6. PREFIX-NORMALIZED COMPARISON")
    print("-" * 88)

    ckpt_norm = {}
    model_norm = {}
    for k in ckpt_sd:
        ckpt_norm.setdefault(strip_prefix(k), []).append(k)
    for k in model_sd:
        model_norm.setdefault(strip_prefix(k), []).append(k)

    norm_common = sorted(set(ckpt_norm) & set(model_norm))
    norm_shape_matches = []
    norm_shape_mismatches = []

    for nk in norm_common:
        ck = ckpt_norm[nk][0]
        mk = model_norm[nk][0]
        if tuple(ckpt_sd[ck].shape) == tuple(model_sd[mk].shape):
            norm_shape_matches.append((ck, mk))
        else:
            norm_shape_mismatches.append((ck, mk))

    print(f"Normalized common keys  : {len(norm_common)}")
    print(f"Normalized shape match  : {len(norm_shape_matches)}")
    print(f"Normalized shape errors : {len(norm_shape_mismatches)}")

    summary["normalized_common_keys"] = len(norm_common)
    summary["normalized_shape_matches"] = len(norm_shape_matches)
    summary["normalized_shape_mismatches"] = len(norm_shape_mismatches)

    # -------------------------------------------------------------------------
    # Architecture fingerprints
    # -------------------------------------------------------------------------
    print("\n7. ARCHITECTURE FINGERPRINT")
    print("-" * 88)

    ckpt_prefixes = Counter(k.split(".")[0] for k in ckpt_sd)
    model_prefixes = Counter(k.split(".")[0] for k in model_sd)

    print("Checkpoint top-level prefixes:")
    for k, n in ckpt_prefixes.most_common(40):
        print(f"  {k}: {n}")

    print("\nCurrent model top-level prefixes:")
    for k, n in model_prefixes.most_common(40):
        print(f"  {k}: {n}")

    def keyword_counts(keys):
        text = "\n".join(keys).lower()
        words = [
            "swin",
            "vit",
            "transformer",
            "encoder",
            "decoder",
            "out",
            "head",
            "conv",
            "norm",
            "attention",
            "patch",
            "relative_position",
            "window",
            "down",
            "up",
        ]
        return {w: text.count(w) for w in words}

    summary["checkpoint_prefixes"] = dict(ckpt_prefixes)
    summary["model_prefixes"] = dict(model_prefixes)
    summary["checkpoint_keyword_counts"] = keyword_counts(ckpt_sd.keys())
    summary["model_keyword_counts"] = keyword_counts(model_sd.keys())

    print("\nCheckpoint keywords:")
    print(summary["checkpoint_keyword_counts"])
    print("Current model keywords:")
    print(summary["model_keyword_counts"])

    # -------------------------------------------------------------------------
    # Tensor-shape signatures
    # -------------------------------------------------------------------------
    print("\n8. SHAPE SIGNATURE COMPARISON")
    print("-" * 88)

    ckpt_shapes = Counter(tuple(v.shape) for v in ckpt_sd.values())
    model_shapes = Counter(tuple(v.shape) for v in model_sd.values())

    shared_shapes = set(ckpt_shapes) & set(model_shapes)
    weighted_shared = sum(
        min(ckpt_shapes[s], model_shapes[s]) for s in shared_shapes
    )

    print(f"Unique checkpoint shapes : {len(ckpt_shapes)}")
    print(f"Unique model shapes      : {len(model_shapes)}")
    print(f"Shared unique shapes     : {len(shared_shapes)}")
    print(f"Shared tensor slots      : {weighted_shared}")

    summary["checkpoint_unique_shapes"] = len(ckpt_shapes)
    summary["model_unique_shapes"] = len(model_shapes)
    summary["shared_unique_shapes"] = len(shared_shapes)
    summary["shared_tensor_slots"] = weighted_shared

    # -------------------------------------------------------------------------
    # Safe strict load
    # -------------------------------------------------------------------------
    print("\n9. STRICT LOAD TEST")
    print("-" * 88)

    strict_possible = (
        len(missing) == 0
        and len(unexpected) == 0
        and len(shape_mismatches) == 0
    )

    if strict_possible:
        try:
            model.load_state_dict(ckpt_sd, strict=True)
            print("Strict load : PASS")
            summary["strict_load"] = "PASS"
        except Exception as e:
            print(f"Strict load : FAIL — {type(e).__name__}: {e}")
            summary["strict_load"] = f"FAIL: {type(e).__name__}: {e}"
    else:
        print("Strict load : NOT ATTEMPTED")
        print("Reason      : key/shape incompatibility detected.")
        summary["strict_load"] = "NOT_ATTEMPTED"

    # -------------------------------------------------------------------------
    # Diagnosis
    # -------------------------------------------------------------------------
    print("\n10. DIAGNOSIS")
    print("-" * 88)

    diagnosis = []

    if len(missing) or len(unexpected):
        diagnosis.append(
            "The checkpoint and current factory have different state-dict key sets."
        )

    if len(shape_mismatches):
        diagnosis.append(
            "Some identically named tensors have different shapes."
        )

    if len(norm_shape_matches) == len(ckpt_sd) and len(norm_shape_matches) == len(model_sd):
        diagnosis.append(
            "The mismatch is fully explained by key prefixes/wrappers."
        )
    elif len(norm_shape_matches) < len(ckpt_sd) * 0.5:
        diagnosis.append(
            "The mismatch is NOT explained by a simple module/model prefix."
        )
    else:
        diagnosis.append(
            "Some compatibility can be explained by prefixes, but the architectures still differ."
        )

    if tensor_numel(ckpt_sd) != all_params:
        diagnosis.append(
            "Checkpoint tensor-element count differs from current model parameter count."
        )

    ratio = tensor_numel(ckpt_sd) / all_params if all_params else None
    summary["checkpoint_to_current_parameter_ratio"] = ratio

    if ratio is not None:
        print(f"Checkpoint/current parameter ratio: {ratio:.6f}")

    if ratio is not None and ratio < 0.5:
        diagnosis.append(
            "The checkpoint is substantially smaller than the current factory model."
        )
    elif ratio is not None and ratio > 1.5:
        diagnosis.append(
            "The checkpoint is substantially larger than the current factory model."
        )

    # Identify whether the checkpoint metadata itself describes the expected
    # training configuration.
    for key in ("feature_size", "num_classes", "full_shape", "crop_shape"):
        if key in metadata:
            print(f"Checkpoint {key}: {metadata[key]}")

    summary["diagnosis"] = diagnosis
    for d in diagnosis:
        print("• " + d)

    if strict_possible:
        final_status = "PASS — CURRENT_FACTORY_MATCHES_FINAL_CHECKPOINT"
    else:
        final_status = "REVIEW_REQUIRED — FACTORY_CHECKPOINT_ARCHITECTURE_MISMATCH"

    summary["final_status"] = final_status

    # -------------------------------------------------------------------------
    # Key CSV
    # -------------------------------------------------------------------------
    rows = []
    for key in sorted(ckpt_keys | model_keys):
        c = ckpt_sd.get(key)
        m = model_sd.get(key)

        if c is None:
            status = "MISSING_IN_CHECKPOINT"
        elif m is None:
            status = "UNEXPECTED_IN_CHECKPOINT"
        elif tuple(c.shape) != tuple(m.shape):
            status = "SHAPE_MISMATCH"
        else:
            status = "MATCH"

        rows.append(
            {
                "key": key,
                "status": status,
                "checkpoint_shape": list(c.shape) if c is not None else "",
                "model_shape": list(m.shape) if m is not None else "",
                "checkpoint_numel": int(c.numel()) if c is not None else "",
                "model_numel": int(m.numel()) if m is not None else "",
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "status",
                "checkpoint_shape",
                "model_shape",
                "checkpoint_numel",
                "model_numel",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "PART 107.1 — CORRECTED MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY",
        "=" * 88,
        f"Project root: {PROJECT_ROOT}",
        f"Python: {sys.executable}",
        f"PyTorch: {torch.__version__}",
        f"CUDA: {torch.cuda.is_available()}",
        f"Checkpoint SHA256: {sha}",
        f"Checkpoint keys: {len(ckpt_sd)}",
        f"Checkpoint tensor elements: {tensor_numel(ckpt_sd):,}",
        f"Factory module: {summary['factory_module']}",
        f"Factory: {factory.__name__}",
        f"Factory call: {call}",
        f"Model class: {summary['model_class']}",
        f"Model parameters: {all_params:,}",
        f"Model state keys: {len(model_sd)}",
        "",
        "COMPATIBILITY",
        "-" * 88,
        f"Common keys: {len(common)}",
        f"Shape-compatible keys: {len(shape_matches)}",
        f"Shape mismatches: {len(shape_mismatches)}",
        f"Missing keys: {len(missing)}",
        f"Unexpected keys: {len(unexpected)}",
        f"Normalized shape matches: {len(norm_shape_matches)}",
        f"Strict load: {summary['strict_load']}",
        "",
        "DIAGNOSIS",
        "-" * 88,
    ] + [f"- {d}" for d in diagnosis] + [
        "",
        "FINAL STATUS",
        "-" * 88,
        final_status,
        "",
        "No training was performed and the final checkpoint was not modified.",
    ]

    TXT_PATH.write_text("\n".join(report), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print("\n========================================================================================")
    print("PART 107.1 FINAL RESULT")
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
