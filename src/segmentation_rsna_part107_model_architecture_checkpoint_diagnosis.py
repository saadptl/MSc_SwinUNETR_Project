"""
PART 107 — MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY DIAGNOSIS

Purpose
-------
Diagnose, without training or modifying any checkpoint, why the selected
Part 104 final segmentation checkpoint does not cleanly load into the current
src.model.create_model() architecture.

This script:
- preserves the selected final checkpoint untouched;
- inspects checkpoint/state-dict metadata and tensor shapes;
- imports the project's current model factory;
- constructs the current model;
- compares checkpoint keys/shapes against model keys/shapes;
- detects common wrapper/prefix differences;
- records model parameter counts using actual model parameters;
- inspects available project configuration/training artifacts;
- checks whether the checkpoint resembles the Part 98 / Part 99 model family;
- produces CSV + JSON + TXT reports.

IMPORTANT
---------
This is an AUDIT ONLY. It does NOT train, overwrite checkpoints, edit src/model.py,
or install packages.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part107_model_architecture_checkpoint_diagnosis"
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

CSV_PATH = OUTPUT_DIR / "part107_key_compatibility.csv"
JSON_PATH = REPORT_DIR / "part107_model_architecture_checkpoint_diagnosis_summary.json"
TXT_PATH = REPORT_DIR / "part107_model_architecture_checkpoint_diagnosis_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "numel": int(obj.numel()),
        }
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def load_checkpoint(path: Path):
    # weights_only=False is intentional because this is a local project artifact
    # and we want to inspect all metadata where available.
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
    best_score = -1
    for name, candidate in candidates:
        tensor_items = [(k, v) for k, v in candidate.items() if isinstance(k, str) and torch.is_tensor(v)]
        score = len(tensor_items)
        if score > best_score:
            best_score = score
            best_name, best = name, candidate

    if best is None:
        raise RuntimeError("No tensor state_dict-like object found in checkpoint.")

    return best_name, {k: v for k, v in best.items() if isinstance(k, str) and torch.is_tensor(v)}


def strip_common_prefix(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "net.",
        "network.",
        "state_dict.",
        "_orig_mod.",
        "backbone.",
    )
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def normalized_key(key: str) -> str:
    key = strip_common_prefix(key)
    # Common DDP / wrapper artifacts.
    key = re.sub(r"^module\.", "", key)
    return key


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def tensor_count(state_dict: dict) -> int:
    return sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))


def shape_signature(state_dict: dict) -> Counter:
    return Counter(tuple(v.shape) for v in state_dict.values() if torch.is_tensor(v))


def inspect_module_source(module) -> dict:
    result = {}
    try:
        result["file"] = inspect.getsourcefile(module)
    except Exception:
        result["file"] = None
    try:
        result["source"] = inspect.getsource(module)
    except Exception:
        result["source"] = None
    return result


def find_factory():
    errors = []
    try:
        module = importlib.import_module("src.model")
    except Exception as e:
        errors.append(f"src.model import failed: {type(e).__name__}: {e}")
        return None, None, errors

    preferred = [
        "create_model",
        "build_model",
        "get_model",
        "make_model",
    ]
    for name in preferred:
        fn = getattr(module, name, None)
        if callable(fn):
            return module, fn, errors

    public_callables = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and any(x in name.lower() for x in ("model", "build", "create")):
            public_callables.append(name)

    errors.append("No preferred model factory found. Candidates: " + ", ".join(public_callables))
    return module, None, errors


def call_factory(factory):
    """
    Try only non-destructive, low-risk factory calls using common signatures.
    No checkpoint is passed here.
    """
    attempts = [
        ("create_model()", lambda: factory()),
        ("create_model(num_classes=5)", lambda: factory(num_classes=5)),
        ("create_model(in_channels=1, out_channels=5)", lambda: factory(in_channels=1, out_channels=5)),
        (
            "create_model(in_channels=1, out_channels=5, feature_size=12)",
            lambda: factory(in_channels=1, out_channels=5, feature_size=12),
        ),
        (
            "create_model(in_channels=1, out_channels=5, img_size=(64,96,96), feature_size=12)",
            lambda: factory(
                in_channels=1,
                out_channels=5,
                img_size=(64, 96, 96),
                feature_size=12,
            ),
        ),
    ]

    errors = []
    for label, fn in attempts:
        try:
            model = fn()
            if isinstance(model, torch.nn.Module):
                return model, label, errors
            errors.append(f"{label}: returned {type(model).__name__}, not torch.nn.Module")
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
    return None, None, errors


def module_level_clues(module) -> dict:
    clues = {}
    for name in (
        "MODEL_NAME",
        "MODEL_TYPE",
        "MODEL_CONFIG",
        "CONFIG",
        "DEFAULT_CONFIG",
        "FEATURE_SIZE",
        "IMG_SIZE",
        "IN_CHANNELS",
        "OUT_CHANNELS",
        "NUM_CLASSES",
    ):
        if hasattr(module, name):
            try:
                clues[name] = safe_json(getattr(module, name))
            except Exception:
                clues[name] = "<unreadable>"

    try:
        src = inspect.getsource(module)
        clues["source_keywords"] = {
            word: (word.lower() in src.lower())
            for word in (
                "SwinUNETR",
                "swin_unetr",
                "UNETR",
                "timm",
                "feature_size",
                "out_channels",
                "img_size",
                "in_channels",
            )
        }
    except Exception:
        pass
    return clues


def collect_artifact_clues():
    patterns = [
        "outputs/segmentation/rsna_part98*",
        "outputs/segmentation/*part98*",
        "outputs/segmentation/*part99*",
        "reports/*part98*",
        "reports/*part99*",
    ]
    found = []
    for pattern in patterns:
        for p in PROJECT_ROOT.glob(pattern):
            found.append(str(p.relative_to(PROJECT_ROOT)))
    return sorted(set(found))[:500]


def main():
    print("=" * 88)
    print("PART 107 — MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY DIAGNOSIS")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print()

    summary = {
        "part": 107,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_found": CHECKPOINT.exists(),
    }

    print("1. SELECTED FINAL CHECKPOINT")
    print("-" * 88)
    if not CHECKPOINT.exists():
        print("Checkpoint : MISSING")
        summary["final_status"] = "FAIL — FINAL_CHECKPOINT_MISSING"
        JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        print("\nFINAL: FAIL — FINAL_CHECKPOINT_MISSING")
        return 1

    size_mb = CHECKPOINT.stat().st_size / (1024 * 1024)
    sha = sha256_file(CHECKPOINT)
    print(f"Checkpoint : FOUND")
    print(f"Size MB    : {size_mb:.3f}")
    print(f"SHA256     : {sha}")
    summary["checkpoint_size_mb"] = size_mb
    summary["checkpoint_sha256"] = sha

    print("\n2. CHECKPOINT STRUCTURE")
    print("-" * 88)
    try:
        ckpt = load_checkpoint(CHECKPOINT)
        source_name, state = extract_state_dict(ckpt)
        ckpt_type = type(ckpt).__name__
        state_keys = list(state.keys())
        raw_numel = tensor_count(state)
        shape_counts = shape_signature(state)

        print(f"Checkpoint object type : {ckpt_type}")
        print(f"State-dict source      : {source_name}")
        print(f"Tensor keys            : {len(state_keys)}")
        print(f"Tensor parameter count : {raw_numel}")
        print(f"Unique tensor shapes   : {len(shape_counts)}")

        summary["checkpoint_object_type"] = ckpt_type
        summary["state_dict_source"] = source_name
        summary["checkpoint_key_count"] = len(state_keys)
        summary["checkpoint_tensor_numel"] = raw_numel
        summary["checkpoint_unique_shapes"] = len(shape_counts)

        metadata = {}
        if isinstance(ckpt, dict):
            for k, v in ckpt.items():
                if k not in {"state_dict", "model_state_dict", "model", "net", "network", "weights", "model_weights"}:
                    if not isinstance(v, torch.Tensor):
                        metadata[k] = safe_json(v)
        summary["checkpoint_metadata"] = metadata

        print("\nTop-level metadata keys:")
        for k in sorted(metadata.keys())[:80]:
            value = metadata[k]
            rendered = json.dumps(value, ensure_ascii=False)
            if len(rendered) > 300:
                rendered = rendered[:300] + "..."
            print(f"  {k}: {rendered}")

    except Exception as e:
        summary["checkpoint_error"] = f"{type(e).__name__}: {e}"
        summary["final_status"] = "FAIL — CHECKPOINT_INSPECTION_ERROR"
        JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        print(f"Checkpoint inspection failed: {type(e).__name__}: {e}")
        print("\nFINAL: FAIL — CHECKPOINT_INSPECTION_ERROR")
        return 1

    print("\n3. CURRENT PROJECT MODEL FACTORY")
    print("-" * 88)
    module, factory, factory_errors = find_factory()
    summary["factory_errors"] = factory_errors

    if module is None or factory is None:
        print("Factory import/selection : FAIL")
        for e in factory_errors:
            print("  " + e)
        summary["final_status"] = "FAIL — MODEL_FACTORY_UNAVAILABLE"
        JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        print("\nFINAL: FAIL — MODEL_FACTORY_UNAVAILABLE")
        return 1

    module_info = inspect_module_source(module)
    clues = module_level_clues(module)
    print(f"Factory module : {module_info.get('file')}")
    print(f"Factory name   : {factory.__name__}")
    print(f"Factory signature: {inspect.signature(factory)}")
    print("Module-level clues:")
    for k, v in clues.items():
        print(f"  {k}: {v}")
    summary["factory_module"] = module_info.get("file")
    summary["factory_name"] = factory.__name__
    summary["factory_signature"] = str(inspect.signature(factory))
    summary["module_clues"] = clues

    print("\n4. CURRENT MODEL CONSTRUCTION")
    print("-" * 88)
    model, factory_call, construction_errors = call_factory(factory)
    summary["construction_errors"] = construction_errors

    if model is None:
        print("Model construction : FAIL")
        for e in construction_errors:
            print("  " + e)
        summary["final_status"] = "FAIL — MODEL_CONSTRUCTION_FAILED"
        JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        TXT_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")
        print("\nFINAL: FAIL — MODEL_CONSTRUCTION_FAILED")
        return 1

    current_param_count = parameter_count(model)
    current_state = model.state_dict()
    current_key_count = len(current_state)
    current_tensor_count = tensor_count(current_state)

    print(f"Factory call         : {factory_call}")
    print(f"Model class          : {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"Trainable parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"All model parameters : {current_param_count:,}")
    print(f"Model state keys     : {current_key_count}")
    print(f"Model tensor elements: {current_tensor_count:,}")

    summary["factory_call"] = factory_call
    summary["model_class"] = f"{model.__class__.__module__}.{model.__class__.__name__}"
    summary["model_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["model_all_parameters"] = current_param_count
    summary["model_state_key_count"] = current_key_count
    summary["model_state_tensor_numel"] = current_tensor_count

    print("\n5. DIRECT STATE-DICT COMPATIBILITY")
    print("-" * 88)
    ckpt_keys = set(state.keys())
    model_keys = set(current_state.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    common = sorted(model_keys & ckpt_keys)
    shape_mismatches = []
    shape_matches = []

    for k in common:
        a = tuple(state[k].shape)
        b = tuple(current_state[k].shape)
        if a == b:
            shape_matches.append(k)
        else:
            shape_mismatches.append({
                "key": k,
                "checkpoint_shape": list(a),
                "model_shape": list(b),
                "checkpoint_numel": int(state[k].numel()),
                "model_numel": int(current_state[k].numel()),
            })

    print(f"Exact common keys       : {len(common)}")
    print(f"Shape-compatible keys   : {len(shape_matches)}")
    print(f"Shape mismatches        : {len(shape_mismatches)}")
    print(f"Missing model keys      : {len(missing)}")
    print(f"Unexpected checkpoint   : {len(unexpected)}")

    summary["exact_common_keys"] = len(common)
    summary["shape_compatible_keys"] = len(shape_matches)
    summary["shape_mismatch_count"] = len(shape_mismatches)
    summary["missing_key_count"] = len(missing)
    summary["unexpected_key_count"] = len(unexpected)

    print("\nFirst missing keys:")
    for k in missing[:30]:
        print("  " + k)
    print("\nFirst unexpected checkpoint keys:")
    for k in unexpected[:30]:
        print("  " + k)
    print("\nFirst shape mismatches:")
    for item in shape_mismatches[:30]:
        print(
            f"  {item['key']}: checkpoint={item['checkpoint_shape']} "
            f"model={item['model_shape']}"
        )

    print("\n6. PREFIX/NORMALIZATION DIAGNOSTIC")
    print("-" * 88)
    normalized_ckpt = {}
    normalized_model = {}
    for k in state:
        normalized_ckpt.setdefault(normalized_key(k), []).append(k)
    for k in current_state:
        normalized_model.setdefault(normalized_key(k), []).append(k)

    norm_ckpt_keys = set(normalized_ckpt)
    norm_model_keys = set(normalized_model)
    norm_common = sorted(norm_ckpt_keys & norm_model_keys)

    prefix_only_match = []
    prefix_shape_mismatch = []
    for nk in norm_common:
        ck = normalized_ckpt[nk][0]
        mk = normalized_model[nk][0]
        if tuple(state[ck].shape) == tuple(current_state[mk].shape):
            prefix_only_match.append((ck, mk))
        else:
            prefix_shape_mismatch.append((ck, mk))

    print(f"Normalized common keys  : {len(norm_common)}")
    print(f"Normalized shape matches: {len(prefix_only_match)}")
    print(f"Normalized shape errors : {len(prefix_shape_mismatch)}")

    summary["normalized_common_keys"] = len(norm_common)
    summary["normalized_shape_matches"] = len(prefix_only_match)
    summary["normalized_shape_mismatches"] = len(prefix_shape_mismatch)

    print("\n7. KEY-PATTERN / ARCHITECTURE CLUES")
    print("-" * 88)
    ckpt_prefixes = Counter(k.split(".")[0] for k in state.keys())
    model_prefixes = Counter(k.split(".")[0] for k in current_state.keys())

    print("Checkpoint top-level key prefixes:")
    for k, n in ckpt_prefixes.most_common(30):
        print(f"  {k}: {n}")

    print("\nCurrent model top-level key prefixes:")
    for k, n in model_prefixes.most_common(30):
        print(f"  {k}: {n}")

    def key_keyword_counts(keys):
        joined = "\n".join(keys).lower()
        words = [
            "swin",
            "swinvi",
            "vit",
            "transformer",
            "encoder",
            "decoder",
            "up",
            "down",
            "out",
            "head",
            "conv",
            "norm",
            "attention",
            "patch",
            "relative_position",
            "window",
        ]
        return {w: joined.count(w) for w in words}

    summary["checkpoint_key_keyword_counts"] = key_keyword_counts(state.keys())
    summary["model_key_keyword_counts"] = key_keyword_counts(current_state.keys())

    print("\nCheckpoint key keyword counts:")
    print(summary["checkpoint_key_keyword_counts"])
    print("Current model key keyword counts:")
    print(summary["model_key_keyword_counts"])

    print("\n8. PROJECT ARTIFACT CLUES")
    print("-" * 88)
    artifacts = collect_artifact_clues()
    print(f"Relevant artifact paths found: {len(artifacts)}")
    for p in artifacts[:120]:
        print("  " + p)
    summary["relevant_artifacts"] = artifacts

    print("\n9. SAFE LOAD TEST")
    print("-" * 88)
    # Do not modify the checkpoint. Load only into a fresh model instance if
    # exact compatibility exists. Otherwise report the reason and do not force.
    safe_load_possible = len(missing) == 0 and len(unexpected) == 0 and len(shape_mismatches) == 0
    summary["strict_load_possible"] = safe_load_possible

    if safe_load_possible:
        try:
            result = model.load_state_dict(state, strict=True)
            print("Strict state_dict load : PASS")
            print(f"Result                 : {result}")
            summary["strict_load_result"] = "PASS"
        except Exception as e:
            print(f"Strict state_dict load : FAIL ({type(e).__name__}: {e})")
            summary["strict_load_result"] = f"FAIL: {type(e).__name__}: {e}"
    else:
        print("Strict state_dict load : NOT ATTEMPTED")
        print("Reason: key/shape incompatibility was detected.")
        summary["strict_load_result"] = "NOT_ATTEMPTED_DUE_TO_INCOMPATIBILITY"

    print("\n10. DIAGNOSIS")
    print("-" * 88)

    ratio = None
    if current_param_count:
        ratio = raw_numel / current_param_count

    diagnosis = []

    if raw_numel != current_param_count:
        diagnosis.append(
            "Checkpoint tensor-element count differs from current model parameter count."
        )

    if len(missing) or len(unexpected):
        diagnosis.append(
            "Checkpoint and current model use substantially different state-dict key sets."
        )

    if len(shape_mismatches):
        diagnosis.append(
            "At least one common key has a tensor-shape mismatch."
        )

    if len(prefix_only_match) > max(10, len(state) * 0.8):
        diagnosis.append(
            "Most incompatibility may be explained by wrapper/prefix naming differences."
        )
    else:
        diagnosis.append(
            "The mismatch is not explained by a simple module/DDP prefix difference."
        )

    if raw_numel < current_param_count:
        diagnosis.append(
            "The saved checkpoint is substantially smaller than the current factory model."
        )
    elif raw_numel > current_param_count:
        diagnosis.append(
            "The saved checkpoint is substantially larger than the current factory model."
        )

    summary["parameter_ratio_checkpoint_over_current"] = ratio
    summary["diagnosis"] = diagnosis

    for d in diagnosis:
        print("• " + d)

    if safe_load_possible:
        final_status = "PASS — MODEL_FACTORY_MATCHES_FINAL_CHECKPOINT"
    else:
        final_status = "REVIEW_REQUIRED — MODEL_FACTORY_DOES_NOT_MATCH_FINAL_CHECKPOINT"

    summary["final_status"] = final_status

    # CSV: one row per key with compatibility classification.
    rows = []
    all_keys = sorted(set(state) | set(current_state))
    for k in all_keys:
        ck = state.get(k)
        mk = current_state.get(k)
        if ck is None:
            status = "MISSING_IN_CHECKPOINT"
            ck_shape = ""
            model_shape = str(list(mk.shape))
        elif mk is None:
            status = "UNEXPECTED_IN_CHECKPOINT"
            ck_shape = str(list(ck.shape))
            model_shape = ""
        else:
            ck_shape = str(list(ck.shape))
            model_shape = str(list(mk.shape))
            status = "MATCH" if tuple(ck.shape) == tuple(mk.shape) else "SHAPE_MISMATCH"

        rows.append({
            "key": k,
            "status": status,
            "checkpoint_shape": ck_shape,
            "model_shape": model_shape,
            "checkpoint_numel": int(ck.numel()) if ck is not None else "",
            "model_numel": int(mk.numel()) if mk is not None else "",
        })

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

    report_lines = [
        "PART 107 — MODEL ARCHITECTURE & CHECKPOINT COMPATIBILITY DIAGNOSIS",
        "=" * 88,
        f"Project root: {PROJECT_ROOT}",
        f"Python: {sys.executable}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"Checkpoint: {CHECKPOINT}",
        f"Checkpoint SHA256: {sha}",
        f"Checkpoint tensor keys: {len(state)}",
        f"Checkpoint tensor elements: {raw_numel:,}",
        f"Factory module: {module_info.get('file')}",
        f"Factory: {factory.__name__}",
        f"Factory call: {factory_call}",
        f"Model class: {summary['model_class']}",
        f"Model all parameters: {current_param_count:,}",
        f"Model state keys: {current_key_count}",
        f"Exact common keys: {len(common)}",
        f"Shape-compatible keys: {len(shape_matches)}",
        f"Shape mismatches: {len(shape_mismatches)}",
        f"Missing checkpoint keys: {len(missing)}",
        f"Unexpected checkpoint keys: {len(unexpected)}",
        f"Normalized shape matches: {len(prefix_only_match)}",
        f"Strict load possible: {safe_load_possible}",
        "",
        "DIAGNOSIS",
        "-" * 88,
    ] + [f"- {d}" for d in diagnosis] + [
        "",
        "FINAL STATUS",
        "-" * 88,
        final_status,
        "",
        "IMPORTANT: This Part 107 script did not train, edit source files, or overwrite the checkpoint.",
    ]

    TXT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(safe_json(summary), indent=2), encoding="utf-8")

    print("\n========================================================================================")
    print("PART 107 FINAL RESULT")
    print("========================================================================================")
    print(final_status)
    print("\nOUTPUTS:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(TXT_PATH)
    print("=" * 88)

    # A review result is expected if the current factory does not match.
    # Return 0 so PowerShell does not treat the diagnostic itself as a crash.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nUNHANDLED PART 107 ERROR")
        print(type(exc).__name__ + ": " + str(exc))
        traceback.print_exc()
        raise
