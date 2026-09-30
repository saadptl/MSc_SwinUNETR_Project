"""
PART 119 — EXACT PART98 TRAINING DATASET + CROP INSTRUMENTATION AUDIT

Purpose
-------
Read-only instrumentation audit for the established Part98 training pipeline.

This script does NOT:
- train
- call backward()
- call optimizer.step()
- modify checkpoints
- modify project source files

It specifically resolves the Part98 Dataset class from the source and attempts
to reproduce the actual Part98 __getitem__ path so that training-crop density
can be measured directly.

Outputs
-------
outputs/segmentation/rsna_part119_exact_part98_training_crop_instrumentation_audit/
    part119_training_crop_density.csv
    part119_training_crop_density_summary.json
reports/
    part119_exact_part98_training_crop_instrumentation_summary.json
    part119_exact_part98_training_crop_instrumentation_report.txt
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PATHS / CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PART98_PATH = SRC_DIR / "segmentation_rsna_part98_strong_full_cohort_training.py"

TRAIN_COHORT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part98_strong_full_cohort_training"
    / "part98_train_cohort.csv"
)

FINAL_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part119_exact_part98_training_crop_instrumentation_audit"
)

REPORT_DIR = PROJECT_ROOT / "reports"

CSV_PATH = OUT_DIR / "part119_training_crop_density.csv"
SUMMARY_PATH = OUT_DIR / "part119_training_crop_density_summary.json"
REPORT_JSON = REPORT_DIR / "part119_exact_part98_training_crop_instrumentation_summary.json"
REPORT_TXT = REPORT_DIR / "part119_exact_part98_training_crop_instrumentation_report.txt"

SEED = 42
MAX_CASES = 250
NUM_CLASSES = 6
FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
MIN_FOREGROUND_VOXELS = 20

CANONICAL_SHA = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)


# ============================================================================
# BASIC HELPERS
# ============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def normalize_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    for c in candidates:
        for actual in df.columns:
            if c.lower() in str(actual).lower():
                return actual
    return None


# ============================================================================
# PART98 SOURCE AST RESOLUTION
# ============================================================================

def parse_part98() -> Tuple[ast.Module, str]:
    source = PART98_PATH.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    return tree, source


def source_segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def class_name_score(node: ast.ClassDef) -> int:
    name = node.name.lower()
    score = 0
    if name == "dataset":
        score += 100
    if "dataset" in name:
        score += 50
    if "rsna" in name:
        score += 20
    if "seg" in name:
        score += 10
    return score


def resolve_dataset_class(tree: ast.Module, source: str) -> Dict[str, Any]:
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    ranked = sorted(classes, key=class_name_score, reverse=True)

    candidates = []
    for node in ranked:
        text = source_segment(source, node)
        lower = text.lower()

        # Require actual Dataset semantics rather than merely the word Dataset.
        has_getitem = any(
            isinstance(x, ast.FunctionDef) and x.name == "__getitem__"
            for x in node.body
        )
        has_len = any(
            isinstance(x, ast.FunctionDef) and x.name == "__len__"
            for x in node.body
        )

        if "dataset" in node.name.lower() or has_getitem:
            candidates.append(
                {
                    "name": node.name,
                    "score": class_name_score(node),
                    "has_getitem": has_getitem,
                    "has_len": has_len,
                    "source_lines": (
                        getattr(node, "lineno", None),
                        getattr(node, "end_lineno", None),
                    ),
                    "text": text,
                }
            )

    # Prefer the strongest actual dataset class.
    selected = None
    for c in candidates:
        if c["has_getitem"] and c["has_len"]:
            selected = c
            break
    if selected is None and candidates:
        selected = candidates[0]

    return {
        "selected": selected,
        "candidates": [
            {k: v for k, v in c.items() if k != "text"} for c in candidates
        ],
    }


def inspect_dataset_init_getitem(
    tree: ast.Module, source: str, class_name: str
) -> Dict[str, Any]:
    node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            node = n
            break

    if node is None:
        return {"found": False}

    methods = {}
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name in {"__init__", "__getitem__", "__len__"}:
                methods[item.name] = {
                    "line_start": item.lineno,
                    "line_end": getattr(item, "end_lineno", item.lineno),
                    "source": source_segment(source, item),
                }

    return {
        "found": True,
        "class": class_name,
        "methods": methods,
    }


# ============================================================================
# SAFE MODULE IMPORT
# ============================================================================

def import_part98_module():
    """
    Import Part98 using its normal module identity.

    The script is expected to protect executable training code with
    if __name__ == "__main__". If a module does execute training on import,
    this function catches the exception and reports it instead of attempting
    anything else.
    """
    module_name = "part98_exact_audit_import"
    spec = importlib.util.spec_from_file_location(module_name, PART98_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create Part98 import specification.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        raise RuntimeError(
            f"Part98 import raised SystemExit({exc.code!r}); "
            "the training script is not safely importable."
        ) from exc

    return module


# ============================================================================
# TENSOR / SAMPLE EXTRACTION
# ============================================================================

def tensor_candidates(obj: Any, path: str = "root") -> List[Tuple[str, torch.Tensor]]:
    found: List[Tuple[str, torch.Tensor]] = []

    if torch.is_tensor(obj):
        found.append((path, obj))
        return found

    if isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(tensor_candidates(v, f"{path}[{k!r}]"))
        return found

    if isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            found.extend(tensor_candidates(v, f"{path}[{i}]"))
        return found

    return found


def choose_mask_tensor(sample: Any) -> Tuple[str, torch.Tensor]:
    candidates = tensor_candidates(sample)

    if not candidates:
        raise TypeError("Dataset sample contains no torch.Tensor.")

    # Strong preference for names that identify masks/labels.
    scored = []
    for path, tensor in candidates:
        p = path.lower()
        score = 0
        if "mask" in p:
            score += 100
        if "label" in p:
            score += 80
        if "target" in p:
            score += 70

        # Segmentation masks are usually integer-valued or low-cardinality.
        if not torch.is_floating_point(tensor):
            score += 20

        try:
            unique_count = int(torch.unique(tensor).numel())
            if unique_count <= NUM_CLASSES + 2:
                score += 20
        except Exception:
            unique_count = None

        scored.append((score, path, tensor, unique_count))

    scored.sort(key=lambda x: x[0], reverse=True)
    _, path, tensor, _ = scored[0]

    return path, tensor


def choose_image_tensor(sample: Any, mask_path: str) -> Optional[Tuple[str, torch.Tensor]]:
    candidates = tensor_candidates(sample)
    ranked = []

    for path, tensor in candidates:
        if path == mask_path:
            continue

        p = path.lower()
        score = 0
        if "image" in p:
            score += 100
        if "img" in p:
            score += 60
        if "volume" in p:
            score += 30

        if torch.is_floating_point(tensor):
            score += 20

        if tensor.numel() > 10000:
            score += 10

        ranked.append((score, path, tensor))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0], reverse=True)
    _, path, tensor = ranked[0]
    return path, tensor


# ============================================================================
# SHAPE / MASK METRICS
# ============================================================================

def squeeze_mask(mask: torch.Tensor) -> torch.Tensor:
    m = mask.detach().cpu()

    while m.ndim > 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            # If there is a batch dimension, use the first sample.
            m = m[0]

    if m.ndim != 3:
        raise ValueError(f"Expected a 3D mask after squeezing, got {tuple(m.shape)}")

    return m.long()


def resize_mask_nearest(mask: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Generic nearest-neighbor resize used only when the Dataset sample exposes
    a mask that is not already FULL_SHAPE.

    This does not replace the exact Part98 preprocessing when the Dataset
    itself returns its final crop. It is only a fallback for instrumentation.
    """
    m = squeeze_mask(mask).float()[None, None]
    out = torch.nn.functional.interpolate(
        m,
        size=shape,
        mode="nearest",
    )
    return out[0, 0].long()


def compute_center_exact_logic(mask: torch.Tensor) -> Tuple[int, int, int]:
    """
    Reproduce Part98's visible compute_foreground_center logic:
    mean coordinate of mask > 0, rounded to integer.
    """
    coords = torch.nonzero(mask > 0, as_tuple=False)
    if coords.numel() == 0:
        d, h, w = mask.shape
        return d // 2, h // 2, w // 2

    center = coords.float().mean(dim=0)
    return tuple(int(round(float(x))) for x in center.tolist())


def crop_exact_logic(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_shape: Tuple[int, int, int],
    center: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cd, ch, cw = crop_shape
    d, h, w = image.shape

    cz, cy, cx = center

    z0 = max(0, min(cz - cd // 2, d - cd))
    y0 = max(0, min(cy - ch // 2, h - ch))
    x0 = max(0, min(cx - cw // 2, w - cw))

    z1 = z0 + cd
    y1 = y0 + ch
    x1 = x0 + cw

    return (
        image[z0:z1, y0:y1, x0:x1],
        mask[z0:z1, y0:y1, x0:x1],
    )


def class_counts(mask: torch.Tensor) -> Dict[int, int]:
    vals, counts = torch.unique(mask, return_counts=True)
    out = {int(v.item()): int(c.item()) for v, c in zip(vals, counts)}
    return out


def safe_fraction(n: int, d: int) -> float:
    return float(n / d) if d else float("nan")


def summarize_mask(
    mask: torch.Tensor,
    prefix: str,
    full_shape_expected: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    m = squeeze_mask(mask)
    total = int(m.numel())
    fg = int((m > 0).sum().item())
    counts = class_counts(m)

    result = {
        f"{prefix}_shape": list(m.shape),
        f"{prefix}_voxels": total,
        f"{prefix}_fg_voxels": fg,
        f"{prefix}_fg_fraction": safe_fraction(fg, total),
        f"{prefix}_unique_labels": sorted(counts.keys()),
    }

    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        result[f"{prefix}_c{c}_voxels"] = n
        result[f"{prefix}_c{c}_fraction"] = safe_fraction(n, total)

    if full_shape_expected is not None:
        result[f"{prefix}_shape_matches_expected"] = tuple(m.shape) == tuple(
            full_shape_expected
        )

    return result


# ============================================================================
# DETERMINE WHETHER DATASET ALREADY RETURNS A FINAL CROP
# ============================================================================

def detect_sample_geometry(
    sample: Any,
    mask_path: str,
) -> Dict[str, Any]:
    _, mask = choose_mask_tensor(sample)
    m = squeeze_mask(mask)

    image_pair = choose_image_tensor(sample, mask_path)

    return {
        "mask_path": mask_path,
        "mask_shape": list(m.shape),
        "image_path": image_pair[0] if image_pair else None,
        "image_shape": list(image_pair[1].shape) if image_pair else None,
        "looks_like_full_shape": tuple(m.shape) == FULL_SHAPE,
        "looks_like_crop_shape": tuple(m.shape) == CROP_SHAPE,
    }


# ============================================================================
# DATAFRAME / DATASET INSTANTIATION
# ============================================================================

def load_train_cohort() -> pd.DataFrame:
    if not TRAIN_COHORT.exists():
        raise FileNotFoundError(f"Missing Part98 training cohort: {TRAIN_COHORT}")

    df = pd.read_csv(TRAIN_COHORT)
    if df.empty:
        raise ValueError("Part98 training cohort is empty.")
    return df


def dataset_constructor_candidates(dataset_cls: type, df: pd.DataFrame):
    """
    Try signatures without making assumptions about the constructor.

    Important: failed constructor attempts are harmless and no training occurs.
    """
    candidates = [
        ((df,), {}),
        ((df, True), {}),
        ((df, False), {}),
        ((), {"df": df}),
        ((), {"dataframe": df}),
        ((), {"cohort": df}),
        ((), {"train_df": df}),
        ((), {"records": df}),
    ]

    for args, kwargs in candidates:
        try:
            ds = dataset_cls(*args, **kwargs)
            return ds, {
                "args_count": len(args),
                "kwargs": list(kwargs.keys()),
            }
        except Exception as exc:
            yield None, {
                "args_count": len(args),
                "kwargs": list(kwargs.keys()),
                "error": repr(exc),
            }


def instantiate_dataset(dataset_cls: type, df: pd.DataFrame):
    failures = []

    for result in dataset_constructor_candidates(dataset_cls, df):
        ds, meta = result
        if ds is not None:
            return ds, meta, failures
        failures.append(meta)

    raise RuntimeError(
        "Could not instantiate the resolved Part98 Dataset class. "
        + json.dumps(failures, default=json_default)
    )


# ============================================================================
# EXACT DATASET SAMPLE AUDIT
# ============================================================================

def audit_dataset(
    dataset: Any,
    cohort: pd.DataFrame,
    max_cases: int = MAX_CASES,
) -> Dict[str, Any]:
    n = min(len(dataset), max_cases)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # We do not use DataLoader because we want one deterministic __getitem__
    # call per selected training row and no worker-side randomness.
    for idx in range(n):
        started = time.time()

        try:
            sample = dataset[idx]

            mask_path, mask_tensor_raw = choose_mask_tensor(sample)
            mask = squeeze_mask(mask_tensor_raw)

            image_pair = choose_image_tensor(sample, mask_path)

            # If the Dataset already returns the exact training crop, measure
            # it directly. Otherwise, if it returns FULL_SHAPE, reproduce the
            # exact Part98 center/crop logic.
            sample_geometry = detect_sample_geometry(sample, mask_path)

            source_stage = "dataset_return"
            crop_mask = mask
            full_mask = None

            if tuple(mask.shape) == FULL_SHAPE:
                full_mask = mask
                center = compute_center_exact_logic(full_mask)

                if image_pair is not None:
                    image = image_pair[1]
                    while image.ndim > 3:
                        if image.shape[0] == 1:
                            image = image[0]
                        elif image.shape[-1] == 1:
                            image = image[..., 0]
                        else:
                            image = image[0]
                    image = image.float()

                    if tuple(image.shape) == FULL_SHAPE:
                        _, crop_mask = crop_exact_logic(
                            image,
                            full_mask,
                            CROP_SHAPE,
                            center,
                        )
                        source_stage = "exact_part98_crop_reproduced"
                    else:
                        _, crop_mask = crop_exact_logic(
                            torch.zeros_like(full_mask, dtype=torch.float32),
                            full_mask,
                            CROP_SHAPE,
                            center,
                        )
                        source_stage = "exact_part98_crop_reproduced_mask_only"
                else:
                    _, crop_mask = crop_exact_logic(
                        torch.zeros_like(full_mask, dtype=torch.float32),
                        full_mask,
                        CROP_SHAPE,
                        center,
                    )
                    source_stage = "exact_part98_crop_reproduced_mask_only"
            elif tuple(mask.shape) == CROP_SHAPE:
                source_stage = "dataset_already_returns_crop"
            else:
                # This is intentionally not silently treated as exact.
                # Report the geometry and fail the row so that we do not invent
                # an unverified preprocessing path.
                raise ValueError(
                    f"Unexpected dataset mask geometry {tuple(mask.shape)}; "
                    f"expected FULL_SHAPE={FULL_SHAPE} or CROP_SHAPE={CROP_SHAPE}."
                )

            total_crop = int(crop_mask.numel())
            fg_crop = int((crop_mask > 0).sum().item())
            ccrop = class_counts(crop_mask)

            result = {
                "dataset_index": idx,
                "source_stage": source_stage,
                "mask_path": mask_path,
                "elapsed_sec": time.time() - started,
                "full_shape": list(full_mask.shape) if full_mask is not None else None,
                "crop_shape": list(crop_mask.shape),
                "crop_voxels": total_crop,
                "crop_fg_voxels": fg_crop,
                "crop_fg_fraction": safe_fraction(fg_crop, total_crop),
                "crop_has_fg": bool(fg_crop > 0),
                "crop_has_min20_fg": bool(fg_crop >= MIN_FOREGROUND_VOXELS),
                "crop_unique_labels": sorted(ccrop.keys()),
            }

            for c in range(NUM_CLASSES):
                nc = ccrop.get(c, 0)
                result[f"crop_c{c}_voxels"] = nc
                result[f"crop_c{c}_fraction"] = safe_fraction(nc, total_crop)
                result[f"crop_has_c{c}"] = bool(nc > 0)

            if full_mask is not None:
                total_full = int(full_mask.numel())
                fg_full = int((full_mask > 0).sum().item())
                result["full_fg_voxels"] = fg_full
                result["full_fg_fraction"] = safe_fraction(fg_full, total_full)
                result["crop_fg_enrichment_vs_full"] = (
                    safe_fraction(fg_crop, total_crop)
                    / safe_fraction(fg_full, total_full)
                    if fg_full > 0
                    else float("nan")
                )

                full_counts = class_counts(full_mask)
                for c in range(NUM_CLASSES):
                    nf = full_counts.get(c, 0)
                    result[f"full_c{c}_voxels"] = nf
                    result[f"full_c{c}_fraction"] = safe_fraction(nf, total_full)
                    result[f"crop_retains_c{c}"] = bool(
                        ccrop.get(c, 0) > 0
                    )

            # Preserve identifying cohort fields without assuming exact names.
            if idx < len(cohort):
                row = cohort.iloc[idx]
                for col in cohort.columns:
                    key = str(col)
                    if any(
                        token in key.lower()
                        for token in (
                            "study",
                            "series",
                            "patient",
                            "condition",
                            "level",
                            "modality",
                        )
                    ):
                        result[f"cohort_{key}"] = normalize_scalar(row[col])

            rows.append(result)

            # Free tensors before next sample.
            del sample
            del mask_tensor_raw

        except Exception as exc:
            failures.append(
                {
                    "dataset_index": idx,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    return {
        "requested_cases": n,
        "completed_cases": len(rows),
        "failed_cases": len(failures),
        "rows": rows,
        "failures": failures,
    }


# ============================================================================
# SUMMARY
# ============================================================================

def percentile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(q)) if not s.empty else float("nan")


def make_summary(
    df: pd.DataFrame,
    audit: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    source_info: Dict[str, Any],
    import_info: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "status": "PASS" if audit["completed_cases"] > 0 and audit["failed_cases"] == 0 else "PARTIAL_OR_FAILED",
        "part": 119,
        "purpose": "Exact Part98 training dataset and crop instrumentation audit",
        "read_only": True,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_modified": False,
        "project_root": str(PROJECT_ROOT),
        "part98_path": str(PART98_PATH),
        "train_cohort": str(TRAIN_COHORT),
        "part104_checkpoint": str(FINAL_CKPT),
        "part104_sha256": sha256_file(FINAL_CKPT) if FINAL_CKPT.exists() else None,
        "canonical_sha256": CANONICAL_SHA,
        "canonical_sha_match": (
            FINAL_CKPT.exists() and sha256_file(FINAL_CKPT) == CANONICAL_SHA
        ),
        "python": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "source_dataset_resolution": source_info,
        "import_info": import_info,
        "dataset_instantiation": dataset_meta,
        "requested_cases": audit["requested_cases"],
        "completed_cases": audit["completed_cases"],
        "failed_cases": audit["failed_cases"],
        "full_shape": list(FULL_SHAPE),
        "crop_shape": list(CROP_SHAPE),
        "min_foreground_voxels": MIN_FOREGROUND_VOXELS,
    }

    if df.empty:
        return summary

    def add_metric(name: str, col: str):
        if col not in df.columns:
            return
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return
        summary[name] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p25": percentile(s, 0.25),
            "p75": percentile(s, 0.75),
            "p90": percentile(s, 0.90),
            "p95": percentile(s, 0.95),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    add_metric("full_fg_fraction", "full_fg_fraction")
    add_metric("crop_fg_fraction", "crop_fg_fraction")
    add_metric("crop_fg_voxels", "crop_fg_voxels")
    add_metric("crop_fg_enrichment_vs_full", "crop_fg_enrichment_vs_full")

    for threshold in (0, MIN_FOREGROUND_VOXELS, 50, 100, 500):
        col = "crop_fg_voxels"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            summary[f"fraction_crops_fg_ge_{threshold}"] = (
                float((s >= threshold).mean()) if not s.empty else float("nan")
            )

    for c in range(1, NUM_CLASSES):
        col = f"crop_has_c{c}"
        if col in df.columns:
            s = df[col].astype(bool)
            summary[f"class_{c}_crop_presence_fraction"] = float(s.mean())

    if "crop_has_fg" in df.columns:
        summary["crop_any_foreground_fraction"] = float(
            df["crop_has_fg"].astype(bool).mean()
        )

    if "crop_has_min20_fg" in df.columns:
        summary["crop_ge20_foreground_fraction"] = float(
            df["crop_has_min20_fg"].astype(bool).mean()
        )

    # Class voxel totals in sampled crops.
    class_summary = {}
    for c in range(NUM_CLASSES):
        col = f"crop_c{c}_voxels"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            class_summary[str(c)] = {
                "total_voxels": int(s.sum()),
                "mean_voxels_per_crop": float(s.mean()),
                "median_voxels_per_crop": float(s.median()),
                "cases_with_class": int((s > 0).sum()),
                "case_presence_fraction": float((s > 0).mean()),
            }
    summary["crop_class_distribution"] = class_summary

    # Identify whether Dataset already returns the final crop.
    if "source_stage" in df.columns:
        summary["source_stage_counts"] = df["source_stage"].value_counts().to_dict()

    return summary


# ============================================================================
# REPORT
# ============================================================================

def write_report(summary: Dict[str, Any], failures: List[Dict[str, Any]]) -> None:
    diagnosis = "UNDETERMINED"

    if summary.get("completed_cases", 0) > 0:
        any_fg = summary.get("crop_any_foreground_fraction")
        ge20 = summary.get("crop_ge20_foreground_fraction")

        if isinstance(any_fg, (int, float)) and isinstance(ge20, (int, float)):
            if any_fg < 0.50:
                diagnosis = "LOW_FOREGROUND_CROP_COVERAGE"
            elif ge20 < 0.50:
                diagnosis = "INSUFFICIENT_MINIMUM_FOREGROUND_CROP_DENSITY"
            else:
                diagnosis = "FOREGROUND_CROP_COVERAGE_MEASURED"

    summary["primary_diagnosis"] = diagnosis

    if diagnosis == "FOREGROUND_CROP_COVERAGE_MEASURED":
        recommendation = (
            "Use the measured crop distribution as the baseline for a controlled "
            "loss/crop ablation. Do not change architecture, cohort, or multiple "
            "training variables simultaneously."
        )
    elif diagnosis in {
        "LOW_FOREGROUND_CROP_COVERAGE",
        "INSUFFICIENT_MINIMUM_FOREGROUND_CROP_DENSITY",
    }:
        recommendation = (
            "The exact training crop distribution should be treated as a major "
            "training-strategy variable. A controlled crop-sampling intervention "
            "is justified before aggressive loss reweighting."
        )
    else:
        recommendation = (
            "Resolve the dataset instrumentation before retraining. Do not infer "
            "crop-density behavior from this audit."
        )

    summary["recommendation"] = recommendation

    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )

    lines = [
        "=" * 96,
        "PART 119 — EXACT PART98 TRAINING DATASET + CROP INSTRUMENTATION AUDIT",
        "=" * 96,
        "",
        "READ-ONLY — NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP",
        "",
        f"Project root : {PROJECT_ROOT}",
        f"Python       : {sys.executable}",
        f"PyTorch      : {torch.__version__}",
        f"CUDA         : {torch.cuda.is_available()}",
        f"GPU          : {summary.get('gpu')}",
        "",
        "LOCKED CHECKPOINT",
        "-" * 96,
        f"Part104 SHA256 : {summary.get('part104_sha256')}",
        f"Canonical SHA  : {'PASS' if summary.get('canonical_sha_match') else 'FAIL'}",
        "",
        "DATASET RESOLUTION",
        "-" * 96,
        json.dumps(summary.get("source_dataset_resolution"), indent=2, default=json_default),
        "",
        "DATASET INSTANTIATION",
        "-" * 96,
        json.dumps(summary.get("dataset_instantiation"), indent=2, default=json_default),
        "",
        "AUDIT RESULT",
        "-" * 96,
        f"Requested cases : {summary.get('requested_cases')}",
        f"Completed cases : {summary.get('completed_cases')}",
        f"Failed cases    : {summary.get('failed_cases')}",
        "",
        f"Full shape      : {FULL_SHAPE}",
        f"Crop shape      : {CROP_SHAPE}",
        f"Min FG voxels   : {MIN_FOREGROUND_VOXELS}",
        "",
    ]

    for key in (
        "full_fg_fraction",
        "crop_fg_fraction",
        "crop_fg_voxels",
        "crop_fg_enrichment_vs_full",
    ):
        if key in summary:
            lines.append(f"{key}:")
            lines.append(json.dumps(summary[key], indent=2, default=json_default))
            lines.append("")

    for key in (
        "crop_any_foreground_fraction",
        "crop_ge20_foreground_fraction",
        "fraction_crops_fg_ge_50",
        "fraction_crops_fg_ge_100",
        "fraction_crops_fg_ge_500",
    ):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")

    lines.extend(
        [
            "",
            "CROP CLASS PRESENCE",
            "-" * 96,
        ]
    )

    for c in range(NUM_CLASSES):
        key = f"class_{c}_crop_presence_fraction"
        if key in summary:
            lines.append(f"class {c}: {summary[key]}")

    lines.extend(
        [
            "",
            "PRIMARY DIAGNOSIS",
            "-" * 96,
            str(summary.get("primary_diagnosis")),
            "",
            "RECOMMENDATION",
            "-" * 96,
            str(summary.get("recommendation")),
            "",
            "IMPORTANT",
            "-" * 96,
            "This audit measures the established Part98 training path. It does not "
            "establish clinical validity. The project targets are development/pseudo-mask "
            "labels as established in the preceding audits.",
            "",
        ]
    )

    if failures:
        lines.extend(
            [
                "FAILURES",
                "-" * 96,
                json.dumps(failures[:20], indent=2, default=json_default),
                "",
            ]
        )

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    ensure_dirs()
    set_deterministic_seed(SEED)

    print("\n" + "=" * 96)
    print("PART 119 — EXACT PART98 TRAINING DATASET + CROP INSTRUMENTATION AUDIT")
    print("=" * 96)
    print("READ-ONLY — NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print(
        "GPU          : "
        + (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A"
        )
    )

    print("\n" + "=" * 96)
    print("1. CHECK LOCKED ARTIFACTS")
    print("=" * 96)

    checks = {
        "Part98 source": PART98_PATH.exists(),
        "Part98 train cohort": TRAIN_COHORT.exists(),
        "Part104 final checkpoint": FINAL_CKPT.exists(),
    }
    for name, ok in checks.items():
        print(f"{name:32s}: {'PASS' if ok else 'FAIL'}")

    if FINAL_CKPT.exists():
        ckpt_sha = sha256_file(FINAL_CKPT)
        print(f"Part104 SHA256 : {ckpt_sha}")
        print(f"Canonical SHA  : {'PASS' if ckpt_sha == CANONICAL_SHA else 'FAIL'}")

    print("\n" + "=" * 96)
    print("2. RESOLVE ACTUAL PART98 DATASET CLASS")
    print("=" * 96)

    tree, source = parse_part98()
    source_info = resolve_dataset_class(tree, source)
    selected = source_info.get("selected")

    if not selected:
        raise RuntimeError("Could not resolve an actual Dataset class in Part98.")

    dataset_class_name = selected["name"]

    print(f"Selected Dataset class : {dataset_class_name}")
    print(f"Source lines            : {selected['source_lines']}")
    print(f"Has __init__            : {'__init__' in inspect_dataset_init_getitem(tree, source, dataset_class_name).get('methods', {})}")
    print(f"Has __getitem__         : {selected['has_getitem']}")
    print(f"Has __len__             : {selected['has_len']}")

    inspect_info = inspect_dataset_init_getitem(tree, source, dataset_class_name)

    # Print constructor and retrieval signatures/source snippets.
    for method_name in ("__init__", "__len__", "__getitem__"):
        method = inspect_info.get("methods", {}).get(method_name)
        if method:
            print(
                f"\n{method_name} source lines "
                f"{method['line_start']}–{method['line_end']}:"
            )
            snippet = method["source"]
            print(snippet[:8000])

    print("\n" + "=" * 96)
    print("3. IMPORT PART98 MODULE — READ-ONLY")
    print("=" * 96)

    import_info = {
        "attempted": True,
        "completed": False,
        "module": None,
        "error": None,
    }

    try:
        part98 = import_part98_module()
        import_info["completed"] = True
        import_info["module"] = str(part98.__name__)
        print("Part98 import: PASS")
    except Exception as exc:
        import_info["error"] = repr(exc)
        print(f"Part98 import: FAIL — {exc}")

        # We cannot reproduce the exact Dataset without importable definitions.
        summary = {
            "status": "FAILED",
            "part": 119,
            "primary_diagnosis": "PART98_DATASET_IMPORT_FAILED",
            "training_performed": False,
            "source_dataset_resolution": source_info,
            "import_info": import_info,
            "part104_sha256": (
                sha256_file(FINAL_CKPT) if FINAL_CKPT.exists() else None
            ),
            "canonical_sha256": CANONICAL_SHA,
            "canonical_sha_match": (
                FINAL_CKPT.exists()
                and sha256_file(FINAL_CKPT) == CANONICAL_SHA
            ),
        }
        REPORT_JSON.write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        REPORT_TXT.write_text(
            "PART 119 FAILED: Part98 could not be imported safely.\n"
            + traceback.format_exc(),
            encoding="utf-8",
        )
        print("\nPart 119 cannot continue because the exact Dataset class is not importable.")
        return 1

    if not hasattr(part98, dataset_class_name):
        raise RuntimeError(
            f"Resolved Dataset class {dataset_class_name!r} is not present after import."
        )

    dataset_cls = getattr(part98, dataset_class_name)

    print("\n" + "=" * 96)
    print("4. INSTANTIATE EXACT PART98 DATASET")
    print("=" * 96)

    cohort = load_train_cohort()
    print(f"Training cohort rows : {len(cohort)}")
    print(f"Audit limit          : {min(len(cohort), MAX_CASES)}")

    try:
        dataset, dataset_meta, constructor_failures = instantiate_dataset(
            dataset_cls, cohort
        )
        dataset_meta = {
            "completed": True,
            "dataset_class": dataset_class_name,
            "constructor": dataset_meta,
            "constructor_failures_before_success": constructor_failures,
            "dataset_length": len(dataset),
        }
        print("Dataset instantiation: PASS")
        print(f"Dataset length       : {len(dataset)}")
        print(
            "Constructor used     : "
            + json.dumps(dataset_meta["constructor"], default=json_default)
        )
    except Exception as exc:
        dataset_meta = {
            "completed": False,
            "dataset_class": dataset_class_name,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"Dataset instantiation: FAIL — {exc}")

        summary = make_summary(
            pd.DataFrame(),
            {
                "requested_cases": 0,
                "completed_cases": 0,
                "failed_cases": 0,
                "rows": [],
                "failures": [],
            },
            dataset_meta,
            source_info,
            import_info,
        )
        summary["primary_diagnosis"] = "PART98_DATASET_INSTANTIATION_FAILED"
        summary["recommendation"] = (
            "Do not retrain. Resolve the exact Part98 Dataset constructor "
            "and rerun the instrumentation."
        )
        REPORT_JSON.write_text(
            json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        REPORT_TXT.write_text(
            "PART 119 DATASET INSTANTIATION FAILED\n\n"
            + json.dumps(summary, indent=2, default=json_default),
            encoding="utf-8",
        )
        return 1

    print("\n" + "=" * 96)
    print("5. PROBE FIRST DATASET SAMPLE")
    print("=" * 96)

    try:
        first_sample = dataset[0]
        mask_path, mask = choose_mask_tensor(first_sample)
        geometry = detect_sample_geometry(first_sample, mask_path)

        print(json.dumps(geometry, indent=2, default=json_default))

        if geometry["looks_like_crop_shape"]:
            print("Geometry interpretation: Dataset already returns the training crop.")
        elif geometry["looks_like_full_shape"]:
            print("Geometry interpretation: Dataset returns FULL_SHAPE; exact crop logic will be reproduced.")
        else:
            print("Geometry interpretation: unexpected; audit rows will report failures.")

        del first_sample
    except Exception as exc:
        print(f"First-sample probe failed: {exc}")

    print("\n" + "=" * 96)
    print("6. AUDIT EXACT TRAINING-CROP DISTRIBUTION")
    print("=" * 96)
    print("No training, backward pass, optimizer step, or checkpoint write will occur.")

    started = time.time()
    audit = audit_dataset(dataset, cohort, MAX_CASES)
    elapsed = time.time() - started

    print(f"Elapsed seconds : {elapsed:.2f}")
    print(f"Completed cases : {audit['completed_cases']}")
    print(f"Failed cases   : {audit['failed_cases']}")

    if audit["failed_cases"]:
        print("\nFirst failures:")
        for f in audit["failures"][:5]:
            print(f"  index={f['dataset_index']}: {f['error']}")

    result_df = pd.DataFrame(audit["rows"])

    if not result_df.empty:
        result_df.to_csv(CSV_PATH, index=False)

    summary = make_summary(
        result_df,
        audit,
        dataset_meta,
        source_info,
        import_info,
    )

    write_report(summary, audit["failures"])

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("7. SUMMARY")
    print("=" * 96)

    for key in (
        "full_fg_fraction",
        "crop_fg_fraction",
        "crop_fg_voxels",
        "crop_fg_enrichment_vs_full",
    ):
        if key in summary:
            print(f"\n{key}:")
            print(json.dumps(summary[key], indent=2, default=json_default))

    print(
        f"\nAny foreground crop fraction : "
        f"{summary.get('crop_any_foreground_fraction', 'N/A')}"
    )
    print(
        f">=20 FG crop fraction       : "
        f"{summary.get('crop_ge20_foreground_fraction', 'N/A')}"
    )

    print("\nClass crop presence:")
    for c in range(1, NUM_CLASSES):
        print(
            f"  class {c}: "
            f"{summary.get(f'class_{c}_crop_presence_fraction', 'N/A')}"
        )

    print("\n" + "=" * 96)
    print("PART 119 FINAL RESULT")
    print("=" * 96)

    if (
        audit["completed_cases"] > 0
        and audit["failed_cases"] == 0
        and summary.get("canonical_sha_match")
    ):
        print("PASS — EXACT PART98 DATASET/CROP INSTRUMENTATION COMPLETED")
        print(f"Diagnosis: {summary.get('primary_diagnosis')}")
    else:
        print("PARTIAL/FAILED — DO NOT USE THIS AUDIT TO DESIGN RETRAINING YET")

    print("\nOutputs:")
    print(CSV_PATH)
    print(SUMMARY_PATH)
    print(REPORT_JSON)
    print(REPORT_TXT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
