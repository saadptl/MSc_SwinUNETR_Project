"""
PART 116 — RSNA TARGET / LABEL INTEGRITY + LEARNABILITY AUDIT

Purpose
-------
Investigate the pseudo-mask/development-label target distribution that the
Part 115 logit diagnostic showed was not being cleanly separated by the model.

READ-ONLY:
    - no training
    - no optimizer
    - no checkpoint modification
    - no model modification
    - no package installation

Canonical sources:
    - Part99 reconstructed validation cohort
    - Established Part9 / Part11 loading pipeline
    - Existing project training/validation cohort CSVs when available

Audits:
    1. Overall class voxel distribution
    2. Per-case class presence
    3. Class sparsity / imbalance
    4. Foreground/background ratios
    5. Connected-component counts
    6. Connected-component size distribution
    7. Tiny-component frequency
    8. Training vs validation target distribution
    9. Duplicate-study effects
   10. Spatial concentration / bounding-box occupancy
   11. Target-label learnability indicators
   12. Overall diagnosis and retraining recommendation

Scientific limitation:
    These are project pseudo-masks/development labels. This audit does not
    establish clinical ground-truth quality.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

VAL_COHORT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part99_clinical_oriented_finetuning"
    / "part99_validation_cohort_reconstructed.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part116_target_label_integrity_learnability_audit"
)

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUTPUT_DIR / "part116_case_target_integrity_audit.csv"
CLASS_CSV = OUTPUT_DIR / "part116_class_target_distribution_summary.csv"
COMPONENT_CSV = OUTPUT_DIR / "part116_component_size_summary.csv"
COHORT_CSV = OUTPUT_DIR / "part116_cohort_distribution_summary.csv"

SUMMARY_JSON = REPORT_DIR / "part116_target_label_integrity_learnability_summary.json"
REPORT_TXT = REPORT_DIR / "part116_target_label_integrity_learnability_report.txt"

NUM_CLASSES = 6
EXPECTED_SHAPE = (64, 96, 96)
VOXELS_PER_CASE = int(np.prod(EXPECTED_SHAPE))
EPS = 1e-8

# Component thresholds in voxels. These are diagnostic thresholds only.
TINY_THRESHOLDS = (5, 10, 25, 50, 100)

# Six-neighbour connectivity for 3-D connected components.
NEIGHBOR_OFFSETS = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
]


def banner(text: str) -> None:
    print()
    print("=" * 92)
    print(text)
    print("=" * 92)


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def percentile(values: List[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def connected_components_6(mask: np.ndarray) -> List[int]:
    """
    Pure NumPy/Python 6-connected component sizes.

    The target masks are only 64x96x96, and this audit is run once over the
    100-case validation cohort, so avoiding a new dependency keeps the
    established environment unchanged.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3-D mask, got {mask.shape}")

    visited = np.zeros(mask.shape, dtype=bool)
    sizes: List[int] = []

    sx, sy, sz = mask.shape

    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                if not mask[x, y, z] or visited[x, y, z]:
                    continue

                stack = [(x, y, z)]
                visited[x, y, z] = True
                count = 0

                while stack:
                    cx, cy, cz = stack.pop()
                    count += 1

                    for dx, dy, dz in NEIGHBOR_OFFSETS:
                        nx = cx + dx
                        ny = cy + dy
                        nz = cz + dz

                        if (
                            0 <= nx < sx
                            and 0 <= ny < sy
                            and 0 <= nz < sz
                            and mask[nx, ny, nz]
                            and not visited[nx, ny, nz]
                        ):
                            visited[nx, ny, nz] = True
                            stack.append((nx, ny, nz))

                sizes.append(count)

    return sizes


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    return (
        int(mins[0]),
        int(maxs[0]),
        int(mins[1]),
        int(maxs[1]),
        int(mins[2]),
        int(maxs[2]),
    )


def bbox_volume(bbox: Optional[Tuple[int, int, int, int, int, int]]) -> int:
    if bbox is None:
        return 0

    x0, x1, y0, y1, z0, z1 = bbox
    return (
        (x1 - x0 + 1)
        * (y1 - y0 + 1)
        * (z1 - z0 + 1)
    )


def load_target(
    row: pd.Series,
    part9: Any,
    part11: Any,
) -> np.ndarray:
    loaded = part11.load_tensor_case(row, part9)

    if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
        raise RuntimeError("Unexpected Part11 loader return.")

    target = loaded[1]

    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()
    else:
        target = np.asarray(target)

    target = np.squeeze(target).astype(np.int16, copy=False)

    if target.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Target shape {target.shape} != {EXPECTED_SHAPE}"
        )

    return target


def find_cohort_candidates() -> List[Path]:
    candidates = []

    for candidate in (ROOT / "outputs").rglob("*.csv"):
        try:
            header = pd.read_csv(candidate, nrows=0)
        except Exception:
            continue

        cols = set(map(str, header.columns))

        if {
            "study_id",
            "series_id",
            "pseudo_mask_path",
        }.issubset(cols):
            candidates.append(candidate)

    def score(path: Path) -> int:
        s = str(path).lower()
        score_value = 0

        for token, weight in [
            ("train", 10),
            ("training", 10),
            ("val", 8),
            ("validation", 8),
            ("cohort", 5),
            ("part99", 4),
            ("part11", 2),
            ("part9", 2),
        ]:
            if token in s:
                score_value += weight

        if "part116" in s:
            score_value -= 30

        return score_value

    return sorted(
        set(candidates),
        key=score,
        reverse=True,
    )


def choose_train_and_val_cohorts(
    validation_df: pd.DataFrame,
    validation_path: Path,
) -> Tuple[pd.DataFrame, Path, Optional[pd.DataFrame], Optional[Path]]:
    """
    Find a matching training cohort and return validation + optional training.

    Validation is explicitly anchored to Part99. Training discovery is
    conservative: a candidate must contain the same target columns and have
    substantially more unique study/series pairs than validation.
    """
    candidates = find_cohort_candidates()

    val_pairs = set(
        zip(
            validation_df["study_id"].astype(str),
            validation_df["series_id"].astype(str),
        )
    )

    train_candidates = []

    for p in candidates:
        if p.resolve() == validation_path.resolve():
            continue

        try:
            d = pd.read_csv(p)
        except Exception:
            continue

        if d.empty:
            continue

        pairs = set(
            zip(
                d["study_id"].astype(str),
                d["series_id"].astype(str),
            )
        )

        overlap = len(pairs & val_pairs)

        # A likely training cohort should be larger and not simply another
        # copy of the 100-row validation cohort.
        if len(pairs) > len(val_pairs) * 2 and overlap < max(
            5,
            int(0.10 * len(val_pairs)),
        ):
            train_candidates.append((p, d))

    if not train_candidates:
        return validation_df, validation_path, None, None

    # Prefer the largest unique-pair candidate.
    train_candidates.sort(
        key=lambda item: len(
            set(
                zip(
                    item[1]["study_id"].astype(str),
                    item[1]["series_id"].astype(str),
                )
            )
        ),
        reverse=True,
    )

    train_path, train_df = train_candidates[0]

    return validation_df, validation_path, train_df, train_path


def main() -> int:
    banner("PART 116 — TARGET / LABEL INTEGRITY + LEARNABILITY AUDIT")

    print(f"Project root : {ROOT}")
    print(f"Validation   : {VAL_COHORT}")

    banner("1. INPUT VALIDATION")

    if not VAL_COHORT.exists():
        print("Part99 validation cohort: MISSING")
        print("\nFINAL STATUS: FAIL — VALIDATION_COHORT_MISSING")
        return 1

    part9_path = (
        ROOT
        / "src"
        / "segmentation_rsna_part9_3d_dataset_loader.py"
    )
    part11_path = (
        ROOT
        / "src"
        / "segmentation_rsna_part11_controlled_pilot_training.py"
    )

    if not part9_path.exists() or not part11_path.exists():
        print("Established Part9/Part11 loader source missing.")
        print("\nFINAL STATUS: FAIL —_ESTABLISHED_LOADER_MISSING")
        return 1

    validation_df = pd.read_csv(VAL_COHORT)

    if validation_df.empty:
        print("Validation cohort is empty.")
        print("\nFINAL STATUS: FAIL —_EMPTY_VALIDATION_COHORT")
        return 1

    print(f"Validation rows : {len(validation_df)}")
    print(
        "Validation unique study/series pairs : "
        f"{len(set(zip(validation_df.study_id.astype(str), validation_df.series_id.astype(str))))}"
    )

    banner("2. ESTABLISHED LOADER")

    part9 = import_module_from_path(
        "part116_part9",
        part9_path,
    )
    part11 = import_module_from_path(
        "part116_part11",
        part11_path,
    )

    print("Part9 loader : FOUND")
    print("Part11 loader: FOUND")
    print(
        "Loader contract: load_tensor_case(row, part9) -> "
        "(image, mask, info)"
    )

    banner("3. TARGET MASK EXTRACTION")

    start = time.time()

    rows: List[Dict[str, Any]] = []
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_case_presence = np.zeros(NUM_CLASSES, dtype=np.int64)

    all_component_sizes: Dict[int, List[int]] = {
        c: [] for c in range(1, NUM_CLASSES)
    }

    failures = []

    for idx, (_, row) in enumerate(
        validation_df.iterrows(),
        start=1,
    ):
        try:
            target = load_target(row, part9, part11)

            unique_values = set(
                np.unique(target).astype(int).tolist()
            )

            invalid_values = sorted(
                v for v in unique_values
                if v < 0 or v >= NUM_CLASSES
            )

            if invalid_values:
                raise RuntimeError(
                    f"Invalid target labels: {invalid_values}"
                )

            counts = np.bincount(
                target.reshape(-1).astype(np.int64),
                minlength=NUM_CLASSES,
            )[:NUM_CLASSES]

            class_total += counts

            for c in range(NUM_CLASSES):
                if counts[c] > 0:
                    class_case_presence[c] += 1

            foreground = target > 0
            fg_count = int(foreground.sum())

            bbox = mask_bbox(foreground)
            bbvol = bbox_volume(bbox)

            class_components = {}
            total_components = 0

            for c in range(1, NUM_CLASSES):
                sizes = connected_components_6(target == c)
                all_component_sizes[c].extend(sizes)

                class_components[c] = len(sizes)
                total_components += len(sizes)

            case = {
                "row_position": idx,
                "study_id": row.get("study_id"),
                "series_id": row.get("series_id"),
                "series_description": row.get(
                    "series_description",
                    "",
                ),
                "total_voxels": VOXELS_PER_CASE,
                "foreground_voxels": fg_count,
                "foreground_fraction": fg_count / VOXELS_PER_CASE,
                "background_voxels": int(counts[0]),
                "background_fraction": counts[0] / VOXELS_PER_CASE,
                "number_present_classes": int(
                    np.sum(counts[1:] > 0)
                ),
                "total_foreground_components": total_components,
                "foreground_bbox_volume": bbvol,
                "foreground_bbox_fraction": (
                    bbvol / VOXELS_PER_CASE
                    if bbvol > 0
                    else 0.0
                ),
                "foreground_bbox_compactness": (
                    fg_count / (bbvol + EPS)
                    if bbvol > 0
                    else 0.0
                ),
                "invalid_label_count": len(invalid_values),
            }

            if bbox is not None:
                (
                    x0,
                    x1,
                    y0,
                    y1,
                    z0,
                    z1,
                ) = bbox
                case.update(
                    {
                        "bbox_x_min": x0,
                        "bbox_x_max": x1,
                        "bbox_y_min": y0,
                        "bbox_y_max": y1,
                        "bbox_z_min": z0,
                        "bbox_z_max": z1,
                    }
                )
            else:
                case.update(
                    {
                        "bbox_x_min": -1,
                        "bbox_x_max": -1,
                        "bbox_y_min": -1,
                        "bbox_y_max": -1,
                        "bbox_z_min": -1,
                        "bbox_z_max": -1,
                    }
                )

            for c in range(NUM_CLASSES):
                case[f"class_{c}_voxels"] = int(counts[c])
                case[f"class_{c}_fraction"] = (
                    counts[c] / VOXELS_PER_CASE
                )
                case[f"class_{c}_present"] = int(
                    counts[c] > 0
                )

            for c in range(1, NUM_CLASSES):
                sizes = connected_components_6(target == c)

                case[f"class_{c}_components"] = len(sizes)
                case[f"class_{c}_largest_component"] = (
                    max(sizes) if sizes else 0
                )
                case[f"class_{c}_median_component"] = (
                    float(np.median(sizes))
                    if sizes
                    else 0.0
                )
                case[f"class_{c}_tiny_le_5"] = int(
                    sum(s <= 5 for s in sizes)
                )
                case[f"class_{c}_tiny_le_10"] = int(
                    sum(s <= 10 for s in sizes)
                )
                case[f"class_{c}_tiny_le_25"] = int(
                    sum(s <= 25 for s in sizes)
                )
                case[f"class_{c}_tiny_le_50"] = int(
                    sum(s <= 50 for s in sizes)
                )
                case[f"class_{c}_tiny_le_100"] = int(
                    sum(s <= 100 for s in sizes)
                )

            rows.append(case)

            if idx == 1 or idx % 10 == 0:
                print(
                    f"[{idx:03d}/{len(validation_df):03d}] "
                    f"study={row.get('study_id')} "
                    f"series={row.get('series_id')} "
                    f"FG={fg_count:,} "
                    f"classes={int(np.sum(counts[1:] > 0))} "
                    f"components={total_components}"
                )

        except Exception as exc:
            failures.append(
                f"row={idx}, study={row.get('study_id')}, "
                f"series={row.get('series_id')}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"[{idx:03d}/{len(validation_df):03d}] "
                f"FAIL — {type(exc).__name__}: {exc}"
            )

    if not rows:
        print("\nFINAL STATUS: FAIL — NO_TARGET_CASES_LOADED")
        return 1

    case_df = pd.DataFrame(rows)
    case_df.to_csv(CASE_CSV, index=False)

    banner("4. AGGREGATE CLASS DISTRIBUTION")

    total_target_voxels = int(class_total.sum())

    class_rows = []

    print(
        f"{'Class':<8}"
        f"{'Voxels':>16}"
        f"{'Global %':>13}"
        f"{'Cases':>12}"
        f"{'Case %':>13}"
        f"{'Median/case':>16}"
    )

    for c in range(NUM_CLASSES):
        case_values = case_df[
            f"class_{c}_voxels"
        ].to_numpy(dtype=float)

        global_fraction = (
            class_total[c] / total_target_voxels
        )

        presence_fraction = (
            class_case_presence[c] / len(case_df)
        )

        median_per_case = float(
            np.median(case_values)
        )

        print(
            f"{c:<8}"
            f"{int(class_total[c]):>16,}"
            f"{global_fraction * 100:>12.4f}%"
            f"{int(class_case_presence[c]):>12}"
            f"{presence_fraction * 100:>12.2f}%"
            f"{median_per_case:>16.2f}"
        )

        class_rows.append(
            {
                "class_id": c,
                "total_voxels": int(class_total[c]),
                "global_fraction": global_fraction,
                "cases_present": int(class_case_presence[c]),
                "case_presence_fraction": presence_fraction,
                "median_voxels_per_case": median_per_case,
                "mean_voxels_per_case": float(
                    np.mean(case_values)
                ),
                "p25_voxels_per_case": float(
                    np.percentile(case_values, 25)
                ),
                "p75_voxels_per_case": float(
                    np.percentile(case_values, 75)
                ),
                "max_voxels_per_case": int(
                    np.max(case_values)
                ),
            }
        )

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(CLASS_CSV, index=False)

    banner("5. CLASS SPARSITY + IMBALANCE")

    fg_total = int(class_total[1:].sum())
    bg_total = int(class_total[0])

    print(
        f"Background voxels : {bg_total:,} "
        f"({bg_total / total_target_voxels * 100:.4f}%)"
    )
    print(
        f"Foreground voxels : {fg_total:,} "
        f"({fg_total / total_target_voxels * 100:.4f}%)"
    )
    print(
        f"Background / foreground ratio : "
        f"{bg_total / max(fg_total, 1):.3f}x"
    )

    fg_class_totals = class_total[1:]
    smallest_fg = int(np.min(fg_class_totals))
    largest_fg = int(np.max(fg_class_totals))

    print(
        f"Largest foreground class voxels : "
        f"{largest_fg:,}"
    )
    print(
        f"Smallest foreground class voxels: "
        f"{smallest_fg:,}"
    )
    print(
        f"Largest/smallest FG class ratio  : "
        f"{largest_fg / max(smallest_fg, 1):.3f}x"
    )

    banner("6. CASE-LEVEL FOREGROUND SPARSITY")

    fg_fraction = case_df[
        "foreground_fraction"
    ].to_numpy(dtype=float)

    print(
        f"Mean FG fraction   : {np.mean(fg_fraction) * 100:.5f}%"
    )
    print(
        f"Median FG fraction : {np.median(fg_fraction) * 100:.5f}%"
    )
    print(
        f"P75 FG fraction    : {np.percentile(fg_fraction, 75) * 100:.5f}%"
    )
    print(
        f"Max FG fraction    : {np.max(fg_fraction) * 100:.5f}%"
    )

    for threshold in [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
    ]:
        count = int(np.sum(fg_fraction < threshold))
        print(
            f"Cases with FG < {threshold * 100:.03f}%: "
            f"{count}/{len(case_df)}"
        )

    banner("7. CONNECTED COMPONENT AUDIT")

    component_rows = []

    for c in range(1, NUM_CLASSES):
        sizes = all_component_sizes[c]

        if sizes:
            arr = np.asarray(sizes, dtype=np.int64)
            tiny_stats = {
                f"components_le_{t}": int(
                    np.sum(arr <= t)
                )
                for t in TINY_THRESHOLDS
            }

            row = {
                "class_id": c,
                "total_components": int(len(arr)),
                "cases_with_component": int(
                    np.sum(
                        case_df[
                            f"class_{c}_components"
                        ].to_numpy()
                        > 0
                    )
                ),
                "mean_components_per_case": float(
                    np.mean(
                        case_df[
                            f"class_{c}_components"
                        ].to_numpy()
                    )
                ),
                "mean_component_size": float(
                    np.mean(arr)
                ),
                "median_component_size": float(
                    np.median(arr)
                ),
                "p25_component_size": float(
                    np.percentile(arr, 25)
                ),
                "p75_component_size": float(
                    np.percentile(arr, 75)
                ),
                "largest_component": int(
                    np.max(arr)
                ),
            }

            row.update(tiny_stats)

            for t in TINY_THRESHOLDS:
                row[f"fraction_components_le_{t}"] = (
                    float(np.mean(arr <= t))
                )

        else:
            row = {
                "class_id": c,
                "total_components": 0,
                "cases_with_component": 0,
                "mean_components_per_case": 0.0,
                "mean_component_size": 0.0,
                "median_component_size": 0.0,
                "p25_component_size": 0.0,
                "p75_component_size": 0.0,
                "largest_component": 0,
            }

            for t in TINY_THRESHOLDS:
                row[f"components_le_{t}"] = 0
                row[f"fraction_components_le_{t}"] = 0.0

        component_rows.append(row)

        print(
            f"Class {c}: components={row['total_components']}, "
            f"median_size={row['median_component_size']:.2f}, "
            f"largest={row['largest_component']}"
        )

    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(COMPONENT_CSV, index=False)

    banner("8. SPATIAL CONCENTRATION AUDIT")

    bbox_fraction = case_df[
        "foreground_bbox_fraction"
    ].to_numpy(dtype=float)

    compactness = case_df[
        "foreground_bbox_compactness"
    ].to_numpy(dtype=float)

    print(
        f"Median FG bounding-box fraction : "
        f"{np.median(bbox_fraction):.6f}"
    )
    print(
        f"P75 FG bounding-box fraction    : "
        f"{np.percentile(bbox_fraction, 75):.6f}"
    )
    print(
        f"Median FG bbox compactness       : "
        f"{np.median(compactness):.6f}"
    )

    bbox_large = int(
        np.sum(bbox_fraction > 0.50)
    )
    bbox_small = int(
        np.sum(bbox_fraction < 0.05)
    )

    print(
        f"Cases bbox > 50% of volume      : "
        f"{bbox_large}/{len(case_df)}"
    )
    print(
        f"Cases bbox < 5% of volume       : "
        f"{bbox_small}/{len(case_df)}"
    )

    banner("9. TRAINING / VALIDATION COHORT DISCOVERY")

    (
        validation_df,
        validation_path,
        training_df,
        training_path,
    ) = choose_train_and_val_cohorts(
        validation_df,
        VAL_COHORT,
    )

    print(f"Validation cohort : {validation_path}")
    print(
        f"Training cohort   : "
        f"{training_path if training_path else 'NOT_FOUND'}"
    )

    cohort_rows = []

    for name, d in [
        ("validation", validation_df),
        ("training", training_df),
    ]:
        if d is None:
            continue

        print(f"\n{name.upper()} rows: {len(d)}")

        # The training cohort may be too large for a full connected-component
        # audit here. Load masks only to calculate class voxel/presence stats.
        # If the same 100-row validation cohort is encountered, skip duplicate
        # loading.
        if name == "validation":
            subset_case = case_df
            class_vox = class_total.copy()
            present = class_case_presence.copy()
        else:
            class_vox = np.zeros(NUM_CLASSES, dtype=np.int64)
            present = np.zeros(NUM_CLASSES, dtype=np.int64)

            train_failures = 0

            for j, (_, r) in enumerate(d.iterrows(), start=1):
                try:
                    target = load_target(r, part9, part11)

                    counts = np.bincount(
                        target.reshape(-1).astype(np.int64),
                        minlength=NUM_CLASSES,
                    )[:NUM_CLASSES]

                    class_vox += counts
                    present += (counts > 0).astype(np.int64)

                except Exception:
                    train_failures += 1

                if j % 100 == 0:
                    print(
                        f"  training target audit: "
                        f"{j}/{len(d)}"
                    )

            if train_failures:
                print(
                    f"Training target load warnings: "
                    f"{train_failures}"
                )

        total = int(class_vox.sum())

        for c in range(NUM_CLASSES):
            cohort_rows.append(
                {
                    "cohort": name,
                    "rows": len(d),
                    "class_id": c,
                    "total_voxels": int(class_vox[c]),
                    "global_fraction": (
                        class_vox[c] / total
                        if total
                        else 0.0
                    ),
                    "cases_present": int(present[c]),
                    "case_presence_fraction": (
                        present[c] / len(d)
                        if len(d)
                        else 0.0
                    ),
                }
            )

    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(COHORT_CSV, index=False)

    banner("10. LEARNABILITY INDICATORS")

    # Class presence among foreground classes.
    fg_presence = (
        class_case_presence[1:]
        / max(len(case_df), 1)
    )

    rare_classes = [
        c
        for c, p in zip(range(1, NUM_CLASSES), fg_presence)
        if p < 0.25
    ]

    ultra_rare_classes = [
        c
        for c, p in zip(range(1, NUM_CLASSES), fg_presence)
        if p < 0.10
    ]

    classes_with_zero_global = [
        c
        for c in range(1, NUM_CLASSES)
        if class_total[c] == 0
    ]

    component_fragmented_classes = []

    for row in component_rows:
        c = int(row["class_id"])
        if (
            row["total_components"] > 0
            and row["fraction_components_le_25"] >= 0.75
        ):
            component_fragmented_classes.append(c)

    print(
        f"Foreground classes present globally: "
        f"{[c for c in range(1, NUM_CLASSES) if class_total[c] > 0]}"
    )
    print(
        f"Rare classes (<25% cases)       : {rare_classes}"
    )
    print(
        f"Ultra-rare classes (<10% cases) : "
        f"{ultra_rare_classes}"
    )
    print(
        f"Globally absent classes          : "
        f"{classes_with_zero_global}"
    )
    print(
        f"Highly fragmented classes       : "
        f"{component_fragmented_classes}"
    )

    # A conservative learnability flag:
    # - target class exists in <10% cases, OR
    # - target class has <100 voxels globally, OR
    # - >=75% of components are <=25 voxels.
    learnability_concerns = []

    for c in range(1, NUM_CLASSES):
        class_total_c = int(class_total[c])
        presence = float(fg_presence[c - 1])

        comp = next(
            r for r in component_rows
            if int(r["class_id"]) == c
        )

        reasons = []

        if presence < 0.10:
            reasons.append("present_in_<10%_cases")

        if class_total_c < 100:
            reasons.append("fewer_than_100_global_voxels")

        if (
            comp["total_components"] > 0
            and comp["fraction_components_le_25"] >= 0.75
        ):
            reasons.append(">=75%_components_<=25_voxels")

        if reasons:
            learnability_concerns.append(
                {
                    "class_id": c,
                    "presence_fraction": presence,
                    "total_voxels": class_total_c,
                    "reasons": reasons,
                }
            )

    for item in learnability_concerns:
        print(
            f"Class {item['class_id']} concerns: "
            f"{', '.join(item['reasons'])}"
        )

    banner("11. OVERALL TARGET DIAGNOSIS")

    validation_fg_fraction = float(
        class_total[1:].sum() / total_target_voxels
    )

    class_presence_min = float(
        np.min(fg_presence)
    )

    # Primary diagnosis rules intentionally emphasize evidence rather than
    # claiming that pseudo-labels are objectively "wrong".
    if classes_with_zero_global:
        primary_diagnosis = (
            "TARGET_CLASS_ABSENCE"
        )
    elif ultra_rare_classes:
        primary_diagnosis = (
            "SEVERE_TARGET_CLASS_SPARSITY"
        )
    elif learnability_concerns:
        primary_diagnosis = (
            "TARGET_LABEL_LEARNABILITY_CONCERNS"
        )
    elif validation_fg_fraction < 0.001:
        primary_diagnosis = (
            "EXTREME_FOREGROUND_TARGET_SPARSITY"
        )
    else:
        primary_diagnosis = (
            "NO_MAJOR_TARGET_INTEGRITY_FAILURE_DETECTED"
        )

    if (
        len(learnability_concerns) >= 2
        or ultra_rare_classes
        or classes_with_zero_global
    ):
        retraining_recommendation = (
            "TARGET_AUDIT_SUPPORTS_TARGETED_TRAINING_INVESTIGATION"
        )
    elif primary_diagnosis == "EXTREME_FOREGROUND_TARGET_SPARSITY":
        retraining_recommendation = (
            "TARGET_SPARSITY_REQUIRES_TRAINING_STRATEGY_REVIEW"
        )
    else:
        retraining_recommendation = (
            "DO_NOT_CHANGE_TRAINING_BASED_ON_TARGET_AUDIT_ALONE"
        )

    print(
        f"Validation foreground fraction : "
        f"{validation_fg_fraction * 100:.6f}%"
    )
    print(
        f"Minimum FG class case presence : "
        f"{class_presence_min * 100:.3f}%"
    )
    print(
        f"Learnability concern classes   : "
        f"{[x['class_id'] for x in learnability_concerns]}"
    )
    print(
        f"Primary diagnosis              : "
        f"{primary_diagnosis}"
    )
    print(
        f"Retraining recommendation      : "
        f"{retraining_recommendation}"
    )

    banner("12. OUTPUTS")

    elapsed = time.time() - start

    print(f"Case CSV       : {CASE_CSV}")
    print(f"Class CSV      : {CLASS_CSV}")
    print(f"Components CSV : {COMPONENT_CSV}")
    print(f"Cohort CSV     : {COHORT_CSV}")
    print(f"Summary JSON   : {SUMMARY_JSON}")
    print(f"Report TXT     : {REPORT_TXT}")
    print(f"Elapsed        : {elapsed:.2f} sec")
    print(f"Completed      : {len(case_df)}")
    print(f"Failures       : {len(failures)}")

    summary = {
        "part": 116,
        "status": (
            "PASS — TARGET_LABEL_INTEGRITY_LEARNABILITY_AUDIT_COMPLETED"
        ),
        "validation_cohort": str(validation_path),
        "validation_rows": int(len(validation_df)),
        "completed_target_cases": int(len(case_df)),
        "failures": failures,
        "expected_shape": list(EXPECTED_SHAPE),
        "total_target_voxels": total_target_voxels,
        "class_total_voxels": class_total.tolist(),
        "class_case_presence": class_case_presence.tolist(),
        "validation_foreground_fraction": validation_fg_fraction,
        "background_foreground_ratio": (
            bg_total / max(fg_total, 1)
        ),
        "rare_classes_under_25_percent": rare_classes,
        "ultra_rare_classes_under_10_percent": ultra_rare_classes,
        "globally_absent_classes": classes_with_zero_global,
        "fragmented_classes": component_fragmented_classes,
        "learnability_concerns": learnability_concerns,
        "primary_diagnosis": primary_diagnosis,
        "retraining_recommendation": retraining_recommendation,
        "training_cohort": (
            str(training_path)
            if training_path
            else None
        ),
        "training_rows": (
            int(len(training_df))
            if training_df is not None
            else None
        ),
        "outputs": {
            "case_csv": str(CASE_CSV),
            "class_csv": str(CLASS_CSV),
            "component_csv": str(COMPONENT_CSV),
            "cohort_csv": str(COHORT_CSV),
            "summary_json": str(SUMMARY_JSON),
            "report_txt": str(REPORT_TXT),
        },
        "scientific_note": (
            "Targets are project pseudo-masks/development labels. "
            "This audit does not establish clinical expert-groundtruth quality."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = [
        "PART 116 — TARGET / LABEL INTEGRITY + LEARNABILITY AUDIT",
        "",
        f"Status: {summary['status']}",
        f"Validation rows: {len(case_df)}",
        f"Primary diagnosis: {primary_diagnosis}",
        f"Retraining recommendation: {retraining_recommendation}",
        "",
        f"Foreground fraction: {validation_fg_fraction:.8f}",
        f"Background/foreground ratio: "
        f"{bg_total / max(fg_total, 1):.4f}x",
        f"Rare classes (<25% cases): {rare_classes}",
        f"Ultra-rare classes (<10% cases): {ultra_rare_classes}",
        f"Globally absent classes: {classes_with_zero_global}",
        f"Fragmented classes: {component_fragmented_classes}",
        "",
        "Learnability concerns:",
    ]

    if learnability_concerns:
        for item in learnability_concerns:
            report.append(
                f"  Class {item['class_id']}: "
                f"{', '.join(item['reasons'])}"
            )
    else:
        report.append("  None under the conservative diagnostic rules.")

    report += [
        "",
        "Scientific limitation:",
        "Targets are project pseudo-masks/development labels.",
        "This audit does not establish clinical expert-groundtruth quality.",
    ]

    REPORT_TXT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    banner("PART 116 FINAL RESULT")
    print(
        "PASS — TARGET LABEL INTEGRITY + LEARNABILITY AUDIT COMPLETED"
    )
    print(f"Primary diagnosis: {primary_diagnosis}")
    print(
        f"Retraining recommendation: "
        f"{retraining_recommendation}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
