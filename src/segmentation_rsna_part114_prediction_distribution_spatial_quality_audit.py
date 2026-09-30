"""
PART 114 — RSNA PREDICTION DISTRIBUTION + SPATIAL QUALITY AUDIT

Purpose
-------
Audit the predictions produced by Part 113 before making any decision about
retraining.

This is an ANALYSIS-ONLY part:
    - no training
    - no checkpoint modification
    - no model modification
    - no package installation

It reads:
    outputs/segmentation/rsna_part113_robust_loader_contract_and_full_inference/
        part113_validation_case_inference.csv
        predictions/*.npz

and audits:
    1. predicted class distribution
    2. foreground volume / fraction
    3. target pseudo-mask distribution
    4. prediction-vs-target volume ratio
    5. per-class Dice distribution
    6. class presence / collapse indicators
    7. extreme over-segmentation cases
    8. correlation between predicted and target foreground volume
    9. aggregate confusion-style voxel counts
   10. whether predictions are dominated by one class

Important scientific limitation:
    The targets are the project's existing pseudo-masks/development labels.
    These metrics are not clinical expert-ground-truth performance.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part113_robust_loader_contract_and_full_inference"
)

CASE_CSV = INPUT_DIR / "part113_validation_case_inference.csv"
PRED_DIR = INPUT_DIR / "predictions"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part114_prediction_distribution_spatial_quality_audit"
)
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_CSV = OUTPUT_DIR / "part114_class_distribution_summary.csv"
CASE_AUDIT_CSV = OUTPUT_DIR / "part114_case_spatial_quality_audit.csv"
EXTREME_CSV = OUTPUT_DIR / "part114_extreme_oversegmentation_cases.csv"
VOXEL_CSV = OUTPUT_DIR / "part114_voxel_confusion_summary.csv"

SUMMARY_JSON = REPORT_DIR / "part114_prediction_distribution_audit_summary.json"
REPORT_TXT = REPORT_DIR / "part114_prediction_distribution_audit_report.txt"

NUM_CLASSES = 6
EXPECTED_SHAPE = (64, 96, 96)
EPS = 1e-8


def banner(text: str) -> None:
    print()
    print("=" * 88)
    print(text)
    print("=" * 88)


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def nanmedian(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else float("nan")


def percentile(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def load_prediction(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "prediction" not in data:
            raise RuntimeError(
                f"Missing 'prediction' array in {path.name}"
            )
        pred = np.asarray(data["prediction"])

    if pred.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"{path.name}: expected {EXPECTED_SHAPE}, got {pred.shape}"
        )

    if not np.issubdtype(pred.dtype, np.integer):
        pred = pred.astype(np.int16)

    return pred


def class_counts(mask: np.ndarray) -> np.ndarray:
    return np.bincount(
        mask.reshape(-1).astype(np.int64),
        minlength=NUM_CLASSES,
    )[:NUM_CLASSES]


def dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    t = target == class_id
    pc = int(p.sum())
    tc = int(t.sum())

    if pc == 0 and tc == 0:
        return float("nan")

    inter = int(np.logical_and(p, t).sum())
    return (2.0 * inter) / (pc + tc + EPS)


def main() -> int:
    banner("PART 114 — PREDICTION DISTRIBUTION + SPATIAL QUALITY AUDIT")

    print(f"Project root : {ROOT}")
    print(f"Input folder : {INPUT_DIR}")
    print(f"Case CSV     : {CASE_CSV}")
    print(f"Predictions  : {PRED_DIR}")

    banner("1. INPUT VALIDATION")

    if not CASE_CSV.exists():
        print("Case CSV: MISSING")
        print("\nFINAL STATUS: FAIL — PART113_CASE_CSV_MISSING")
        return 1

    if not PRED_DIR.exists():
        print("Prediction directory: MISSING")
        print("\nFINAL STATUS: FAIL — PART113_PREDICTIONS_MISSING")
        return 1

    df = pd.read_csv(CASE_CSV)

    if df.empty:
        print("Case CSV: EMPTY")
        print("\nFINAL STATUS: FAIL — EMPTY_CASE_RESULTS")
        return 1

    print(f"Case result rows : {len(df)}")

    prediction_files = sorted(PRED_DIR.glob("*.npz"))
    print(f"Prediction files : {len(prediction_files)}")

    if len(prediction_files) == 0:
        print("\nFINAL STATUS: FAIL — NO_PREDICTION_FILES")
        return 1

    banner("2. CASE-LEVEL PREDICTION AUDIT")

    rows: List[Dict] = []
    aggregate_pred = np.zeros(NUM_CLASSES, dtype=np.int64)
    aggregate_target = np.zeros(NUM_CLASSES, dtype=np.int64)

    voxel_confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    failures = []

    # Part 113's result CSV intentionally contains inference-result metadata,
    # but it does NOT necessarily retain the pseudo_mask_path column required
    # by the original Part 9 loader. Therefore, recover the canonical source
    # cohort separately and merge it by study_id + series_id.
    #
    # This keeps Part 114 read-only and preserves the exact Part 9/Part 11
    # target-loading contract used by Part 113.
    usable_df = df[df["status"].astype(str).str.upper() == "PASS"].copy()

    print(f"Successful Part113 rows available : {len(usable_df)}")

    required_keys = {"study_id", "series_id"}
    if not required_keys.issubset(usable_df.columns):
        raise RuntimeError(
            "Part113 CSV is missing study_id and/or series_id."
        )

    cohort_candidates = []

    # Search established project CSVs only. We specifically require
    # pseudo_mask_path because that is the canonical Part9 input contract.
    for candidate in (ROOT / "outputs").rglob("*.csv"):
        try:
            header = pd.read_csv(candidate, nrows=0)
        except Exception:
            continue

        cols = set(header.columns.astype(str))
        if (
            "pseudo_mask_path" in cols
            and "study_id" in cols
            and "series_id" in cols
        ):
            cohort_candidates.append(candidate)

    # Prefer files whose names/folders indicate validation/Part9/Part11
    # provenance, while still allowing the script to work if naming differs.
    def cohort_score(p: Path) -> int:
        s = str(p).lower()
        score = 0
        for token, weight in [
            ("validation", 8),
            ("val", 5),
            ("cohort", 5),
            ("part11", 3),
            ("part9", 3),
            ("rsna", 2),
        ]:
            if token in s:
                score += weight
        if "part113" in s:
            score -= 10
        if "part114" in s:
            score -= 20
        return score

    cohort_candidates = sorted(
        set(cohort_candidates),
        key=cohort_score,
        reverse=True,
    )

    if not cohort_candidates:
        raise RuntimeError(
            "Could not locate any project CSV containing the canonical "
            "pseudo_mask_path, study_id, and series_id columns."
        )

    cohort_df = None
    cohort_path = None

    for candidate in cohort_candidates:
        try:
            candidate_df = pd.read_csv(candidate)
            if candidate_df.empty:
                continue

            pairs = set(
                zip(
                    candidate_df["study_id"].astype(str),
                    candidate_df["series_id"].astype(str),
                )
            )

            requested_pairs = set(
                zip(
                    usable_df["study_id"].astype(str),
                    usable_df["series_id"].astype(str),
                )
            )

            overlap = len(pairs & requested_pairs)

            if overlap >= max(1, int(0.80 * len(requested_pairs))):
                cohort_df = candidate_df.copy()
                cohort_path = candidate
                break
        except Exception:
            continue

    if cohort_df is None:
        raise RuntimeError(
            "Found CSVs with pseudo_mask_path, but none matched the Part113 "
            "validation study_id/series_id pairs sufficiently."
        )

    print(f"Canonical target cohort : {cohort_path}")
    print(f"Canonical cohort rows   : {len(cohort_df)}")

    # Build an exact study_id + series_id lookup. This also handles the
    # project's known duplicate study rows without inventing new mappings.
    cohort_lookup = {}
    for _, cohort_row in cohort_df.iterrows():
        key = (
            str(cohort_row["study_id"]),
            str(cohort_row["series_id"]),
        )
        if key not in cohort_lookup:
            cohort_lookup[key] = cohort_row

    for _, source_row in usable_df.iterrows():
        key = (
            str(source_row["study_id"]),
            str(source_row["series_id"]),
        )

        if key not in cohort_lookup:
            failures.append(
                f"study_id={key[0]}, series_id={key[1]}: "
                "not found in canonical target cohort"
            )
            continue

        live_row = cohort_lookup[key]

        # Copy Part113 metadata into the audit row, but use the canonical
        # cohort row for Part9/Part11 loading.
        audit_source_row = source_row

        prediction_rel = str(source_row.get("prediction_file", "")).strip()

        if not prediction_rel:
            failures.append(
                f"row {source_row.get('row_position', '')}: "
                "missing prediction_file"
            )
            continue

        pred_path = ROOT / prediction_rel

        if not pred_path.exists():
            # Also permit a direct basename lookup.
            direct = PRED_DIR / Path(prediction_rel).name
            if direct.exists():
                pred_path = direct
            else:
                failures.append(
                    f"row {source_row.get('row_position', '')}: "
                    f"missing {prediction_rel}"
                )
                continue

        try:
            pred = load_prediction(pred_path)
        except Exception as exc:
            failures.append(
                f"{pred_path.name}: {type(exc).__name__}: {exc}"
            )
            continue

        # Reconstruct target class volumes from the Part113 CSV.
        target_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

        for c in range(1, NUM_CLASSES):
            value = source_row.get(
                f"class_{c}_dice_pseudomask",
                float("nan"),
            )

        # Part 113 stores target foreground voxels but not the target array.
        # We therefore derive the target class counts only where they are
        # represented by Part113 metadata and use prediction arrays for exact
        # prediction counts. Exact target spatial confusion requires the source
        # Part9/Part11 mask, so this part intentionally performs a second,
        # read-only target load from the established pipeline.
        #
        # Importing the established loaders here keeps the audit tied to the
        # same preprocessing contract used in Part113.
        if not hasattr(main, "_pipeline"):
            import importlib.util

            def import_path(name: str, path: Path):
                spec = importlib.util.spec_from_file_location(name, str(path))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module

            part9 = import_path(
                "part114_part9",
                ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py",
            )
            part11 = import_path(
                "part114_part11",
                ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py",
            )
            main._pipeline = (part9, part11)

        part9, part11 = main._pipeline

        # Part113 established the LIVE contract as row,part9.
        # IMPORTANT: use the canonical cohort row here, not the Part113
        # result row, because only the canonical row carries pseudo_mask_path.
        loaded = part11.load_tensor_case(live_row, part9)
        if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
            raise RuntimeError("Unexpected Part11 loader return.")

        target = np.asarray(
            loaded[1].detach().cpu().numpy()
            if hasattr(loaded[1], "detach")
            else loaded[1]
        ).astype(np.int16)

        if target.ndim == 4 and target.shape[0] == 1:
            target = target[0]

        if target.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"Target shape {target.shape} != {EXPECTED_SHAPE}"
            )

        pred_counts = class_counts(pred)
        target_counts = class_counts(target)

        aggregate_pred += pred_counts
        aggregate_target += target_counts

        flat_pred = pred.reshape(-1).astype(np.int64)
        flat_target = target.reshape(-1).astype(np.int64)

        np.add.at(
            voxel_confusion,
            (flat_target, flat_pred),
            1,
        )

        total_voxels = pred.size

        result = {
            "row_position": source_row.get("row_position"),
            "study_id": source_row.get("study_id"),
            "series_id": source_row.get("series_id"),
            "series_description": source_row.get(
                "series_description",
                "",
            ),
            "prediction_file": str(
                pred_path.relative_to(ROOT)
            ),
            "total_voxels": total_voxels,
            "pred_foreground_voxels": int(pred_counts[1:].sum()),
            "target_foreground_voxels": int(target_counts[1:].sum()),
            "pred_foreground_fraction": float(
                pred_counts[1:].sum() / total_voxels
            ),
            "target_foreground_fraction": float(
                target_counts[1:].sum() / total_voxels
            ),
            "foreground_volume_ratio": float(
                pred_counts[1:].sum()
                / (target_counts[1:].sum() + EPS)
            ),
            "background_prediction_fraction": float(
                pred_counts[0] / total_voxels
            ),
            "prediction_entropy": float(
                -sum(
                    (
                        (count / total_voxels)
                        * math.log(
                            count / total_voxels + EPS
                        )
                    )
                    for count in pred_counts
                    if count > 0
                )
            ),
        }

        for c in range(NUM_CLASSES):
            result[f"pred_class_{c}_voxels"] = int(pred_counts[c])
            result[f"target_class_{c}_voxels"] = int(target_counts[c])
            result[f"pred_class_{c}_fraction"] = float(
                pred_counts[c] / total_voxels
            )
            result[f"target_class_{c}_fraction"] = float(
                target_counts[c] / total_voxels
            )

        for c in range(1, NUM_CLASSES):
            result[f"class_{c}_dice"] = dice(
                pred,
                target,
                c,
            )

        predicted_nonzero = pred[pred > 0]
        if predicted_nonzero.size:
            binc = np.bincount(
                predicted_nonzero.astype(np.int64),
                minlength=NUM_CLASSES,
            )
            dominant_class = int(np.argmax(binc[1:]) + 1)
            dominant_count = int(binc[dominant_class])
            dominant_fraction = (
                dominant_count / int(pred_counts[1:].sum())
            )
        else:
            dominant_class = 0
            dominant_fraction = 0.0

        result["dominant_predicted_foreground_class"] = dominant_class
        result["dominant_foreground_class_fraction"] = float(
            dominant_fraction
        )

        # Extreme over-segmentation flag.
        ratio = result["foreground_volume_ratio"]
        result["extreme_oversegmentation"] = bool(
            ratio >= 20.0
        )

        rows.append(result)

    if not rows:
        print("No usable cases remained after audit.")
        print("\nFINAL STATUS: FAIL — NO_USABLE_PREDICTIONS")
        return 1

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(CASE_AUDIT_CSV, index=False)

    banner("3. AGGREGATE CLASS DISTRIBUTION")

    total_voxels = int(aggregate_pred.sum())

    class_rows = []

    print()
    print(
        f"{'Class':<8}"
        f"{'Pred voxels':>16}"
        f"{'Pred %':>12}"
        f"{'Target voxels':>18}"
        f"{'Target %':>12}"
        f"{'Pred/Target':>15}"
    )

    for c in range(NUM_CLASSES):
        p = int(aggregate_pred[c])
        t = int(aggregate_target[c])

        pp = p / total_voxels
        tp = t / total_voxels

        ratio = p / (t + EPS)

        print(
            f"{c:<8}"
            f"{p:>16,}"
            f"{pp * 100:>11.3f}%"
            f"{t:>18,}"
            f"{tp * 100:>11.3f}%"
            f"{ratio:>15.3f}"
        )

        class_rows.append(
            {
                "class_id": c,
                "predicted_voxels": p,
                "predicted_fraction": pp,
                "target_voxels": t,
                "target_fraction": tp,
                "prediction_to_target_ratio": ratio,
            }
        )

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(CLASS_CSV, index=False)

    banner("4. PER-CLASS DICE DISTRIBUTION")

    dice_summary = {}

    for c in range(1, NUM_CLASSES):
        col = f"class_{c}_dice"
        values = audit_df[col].to_numpy(dtype=float)

        present = values[np.isfinite(values)]

        dice_summary[str(c)] = {
            "mean": nanmean(values),
            "median": nanmedian(values),
            "p25": percentile(values, 25),
            "p75": percentile(values, 75),
            "max": float(np.max(present)) if present.size else float("nan"),
            "cases_with_defined_dice": int(present.size),
            "cases_with_positive_dice": int(
                np.sum(present > 0)
            ) if present.size else 0,
        }

        print(
            f"Class {c}: "
            f"mean={dice_summary[str(c)]['mean']:.6f}, "
            f"median={dice_summary[str(c)]['median']:.6f}, "
            f"p75={dice_summary[str(c)]['p75']:.6f}, "
            f"positive_cases="
            f"{dice_summary[str(c)]['cases_with_positive_dice']}"
        )

    banner("5. PREDICTION COLLAPSE / OVER-SEGMENTATION AUDIT")

    predicted_fg = aggregate_pred[1:].sum()
    target_fg = aggregate_target[1:].sum()

    aggregate_fg_ratio = predicted_fg / (target_fg + EPS)

    pred_class_fg = aggregate_pred[1:]
    if predicted_fg > 0:
        dominant_class = int(np.argmax(pred_class_fg) + 1)
        dominant_fraction = (
            int(pred_class_fg[dominant_class - 1])
            / int(predicted_fg)
        )
    else:
        dominant_class = 0
        dominant_fraction = 0.0

    print(
        f"Aggregate predicted foreground voxels : "
        f"{int(predicted_fg):,}"
    )
    print(
        f"Aggregate target foreground voxels    : "
        f"{int(target_fg):,}"
    )
    print(
        f"Aggregate foreground volume ratio     : "
        f"{aggregate_fg_ratio:.3f}x"
    )
    print(
        f"Dominant predicted foreground class   : "
        f"{dominant_class}"
    )
    print(
        f"Dominant class share of predicted FG   : "
        f"{dominant_fraction * 100:.3f}%"
    )

    if aggregate_fg_ratio >= 20:
        overseg_status = "SEVERE_OVERSEGMENTATION"
    elif aggregate_fg_ratio >= 5:
        overseg_status = "MODERATE_OVERSEGMENTATION"
    else:
        overseg_status = "NOT_SEVERE_BY_VOLUME_RATIO"

    if dominant_fraction >= 0.80:
        collapse_status = "STRONG_FOREGROUND_CLASS_DOMINANCE"
    elif dominant_fraction >= 0.60:
        collapse_status = "MODERATE_FOREGROUND_CLASS_DOMINANCE"
    else:
        collapse_status = "NO_SINGLE_CLASS_DOMINANCE"

    print(f"Volume-ratio diagnosis: {overseg_status}")
    print(f"Class-dominance diagnosis: {collapse_status}")

    banner("6. EXTREME CASES")

    extreme_df = audit_df[
        audit_df["extreme_oversegmentation"] == True
    ].copy()

    extreme_df = extreme_df.sort_values(
        "foreground_volume_ratio",
        ascending=False,
    )

    print(
        f"Cases with prediction/target foreground ratio >= 20x: "
        f"{len(extreme_df)}"
    )

    if not extreme_df.empty:
        display_cols = [
            "row_position",
            "study_id",
            "series_id",
            "series_description",
            "pred_foreground_voxels",
            "target_foreground_voxels",
            "foreground_volume_ratio",
            "dominant_predicted_foreground_class",
            "dominant_foreground_class_fraction",
        ]

        print(
            extreme_df[display_cols]
            .head(15)
            .to_string(index=False)
        )

    extreme_df.to_csv(EXTREME_CSV, index=False)

    banner("7. VOXEL CONFUSION SUMMARY")

    voxel_df = pd.DataFrame(
        voxel_confusion,
        index=[
            f"target_{c}" for c in range(NUM_CLASSES)
        ],
        columns=[
            f"pred_{c}" for c in range(NUM_CLASSES)
        ],
    )

    voxel_df.to_csv(VOXEL_CSV)

    print("Target rows → Predicted columns")
    print(voxel_df.to_string())

    banner("8. FOREGROUND VOLUME CORRELATION")

    pred_fg_per_case = audit_df[
        "pred_foreground_voxels"
    ].to_numpy(dtype=float)

    target_fg_per_case = audit_df[
        "target_foreground_voxels"
    ].to_numpy(dtype=float)

    if (
        len(audit_df) >= 2
        and np.std(pred_fg_per_case) > 0
        and np.std(target_fg_per_case) > 0
    ):
        correlation = float(
            np.corrcoef(
                pred_fg_per_case,
                target_fg_per_case,
            )[0, 1]
        )
    else:
        correlation = float("nan")

    print(
        f"Pearson correlation, predicted vs target "
        f"foreground volume: {correlation:.6f}"
    )

    banner("9. OVERALL DIAGNOSIS")

    mean_dice_all = nanmean(
        np.concatenate(
            [
                audit_df[f"class_{c}_dice"].to_numpy(dtype=float)
                for c in range(1, NUM_CLASSES)
            ]
        )
    )

    median_case_fg_ratio = nanmedian(
        audit_df["foreground_volume_ratio"]
    )

    cases_ratio_5 = int(
        np.sum(
            audit_df["foreground_volume_ratio"].to_numpy()
            >= 5
        )
    )

    cases_ratio_20 = int(
        np.sum(
            audit_df["foreground_volume_ratio"].to_numpy()
            >= 20
        )
    )

    if aggregate_fg_ratio >= 20:
        primary_diagnosis = "SEVERE_FOREGROUND_OVERSEGMENTATION"
    elif aggregate_fg_ratio >= 5:
        primary_diagnosis = "MEANINGFUL_FOREGROUND_OVERSEGMENTATION"
    elif dominant_fraction >= 0.80:
        primary_diagnosis = "PREDICTION_CLASS_COLLAPSE"
    elif mean_dice_all < 0.05:
        primary_diagnosis = "VERY_LOW_PSEUDOMASK_OVERLAP"
    else:
        primary_diagnosis = "NO_SINGLE_DOMINANT_FAILURE_PATTERN"

    print(f"Primary diagnosis : {primary_diagnosis}")
    print(f"Mean defined class Dice : {mean_dice_all:.6f}")
    print(f"Median foreground ratio : {median_case_fg_ratio:.3f}x")
    print(f"Cases >= 5x ratio       : {cases_ratio_5}")
    print(f"Cases >= 20x ratio      : {cases_ratio_20}")
    print(f"FG volume correlation   : {correlation:.6f}")

    banner("10. OUTPUTS")

    print(f"Class distribution CSV : {CLASS_CSV}")
    print(f"Case audit CSV         : {CASE_AUDIT_CSV}")
    print(f"Extreme cases CSV      : {EXTREME_CSV}")
    print(f"Voxel confusion CSV    : {VOXEL_CSV}")

    summary = {
        "part": 114,
        "status": "PASS — PREDICTION_DISTRIBUTION_AUDIT_COMPLETED",
        "cohort_rows_in_part113": int(len(df)),
        "usable_prediction_cases": int(len(audit_df)),
        "audit_failures": failures,
        "expected_shape": list(EXPECTED_SHAPE),
        "canonical_target_cohort": str(cohort_path),
        "canonical_target_cohort_rows": int(len(cohort_df)),
        "aggregate_predicted_class_voxels": aggregate_pred.tolist(),
        "aggregate_target_class_voxels": aggregate_target.tolist(),
        "aggregate_foreground_ratio": float(
            aggregate_fg_ratio
        ),
        "dominant_predicted_foreground_class": int(
            dominant_class
        ),
        "dominant_predicted_foreground_fraction": float(
            dominant_fraction
        ),
        "oversegmentation_status": overseg_status,
        "class_dominance_status": collapse_status,
        "primary_diagnosis": primary_diagnosis,
        "mean_defined_class_dice": float(mean_dice_all),
        "median_case_foreground_ratio": float(
            median_case_fg_ratio
        ),
        "cases_foreground_ratio_ge_5": cases_ratio_5,
        "cases_foreground_ratio_ge_20": cases_ratio_20,
        "foreground_volume_pearson_correlation": (
            correlation
        ),
        "per_class_dice_summary": dice_summary,
        "outputs": {
            "class_distribution_csv": str(CLASS_CSV),
            "case_audit_csv": str(CASE_AUDIT_CSV),
            "extreme_cases_csv": str(EXTREME_CSV),
            "voxel_confusion_csv": str(VOXEL_CSV),
            "summary_json": str(SUMMARY_JSON),
            "report_txt": str(REPORT_TXT),
        },
        "scientific_note": (
            "Target masks are the existing project pseudo-masks/development "
            "labels. Metrics are not clinical expert-ground-truth performance."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = [
        "PART 114 — PREDICTION DISTRIBUTION + SPATIAL QUALITY AUDIT",
        "",
        f"Status: {summary['status']}",
        f"Usable cases: {len(audit_df)}",
        f"Primary diagnosis: {primary_diagnosis}",
        f"Mean defined class Dice: {mean_dice_all:.6f}",
        f"Aggregate foreground ratio: {aggregate_fg_ratio:.3f}x",
        f"Median case foreground ratio: {median_case_fg_ratio:.3f}x",
        f"Cases >= 5x: {cases_ratio_5}",
        f"Cases >= 20x: {cases_ratio_20}",
        f"Foreground volume correlation: {correlation:.6f}",
        f"Dominant predicted foreground class: {dominant_class}",
        f"Dominant foreground share: {dominant_fraction:.6f}",
        "",
        "Per-class Dice summary:",
    ]

    for c in range(1, NUM_CLASSES):
        item = dice_summary[str(c)]
        report.append(
            f"  Class {c}: mean={item['mean']:.6f}, "
            f"median={item['median']:.6f}, "
            f"p25={item['p25']:.6f}, "
            f"p75={item['p75']:.6f}"
        )

    report += [
        "",
        "Scientific limitation:",
        "The targets are project pseudo-masks/development labels.",
        "These results must not be reported as clinical expert-ground-truth accuracy.",
        "",
        f"Class CSV: {CLASS_CSV}",
        f"Case audit CSV: {CASE_AUDIT_CSV}",
        f"Extreme cases CSV: {EXTREME_CSV}",
        f"Voxel confusion CSV: {VOXEL_CSV}",
    ]

    REPORT_TXT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    banner("PART 114 FINAL RESULT")
    print("PASS — PREDICTION DISTRIBUTION AUDIT COMPLETED")
    print(f"Primary diagnosis: {primary_diagnosis}")

    if failures:
        print(
            f"Warnings: {len(failures)} case(s) had audit failures."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
