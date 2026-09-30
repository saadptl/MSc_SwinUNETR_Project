"""
PART 115 — RSNA LOGIT / PROBABILITY / CLASS-BIAS DIAGNOSTIC

Purpose
-------
Diagnose WHY the Part 104 final Swin-UNETR checkpoint produces the severe
foreground over-segmentation identified by Part 114.

This is READ-ONLY diagnostic inference:
    - no training
    - no optimizer
    - no checkpoint modification
    - no source-model modification
    - no package installation

The audit reuses:
    - Part 104 final checkpoint
    - Part 113 canonical validation cohort
    - established Part 9 / Part 11 preprocessing
    - exact canonical MONAI SwinUNETR architecture:
          in_channels=1
          out_channels=6
          feature_size=12
          spatial_dims=3
          use_checkpoint=False

It intentionally performs fresh forward passes so we can inspect the raw
logits/probabilities rather than only the argmax predictions saved by Part113.

Diagnostics:
    1. Per-class mean/std/min/max logits
    2. Per-class mean probabilities
    3. Per-class maximum probabilities
    4. Background vs foreground probability mass
    5. Class-1 probability dominance
    6. Argmax prediction fractions
    7. Confidence / entropy
    8. Background-vs-foreground probability margins
    9. Agreement with Part113 predictions
   10. Probability mass on pseudo-mask target classes
   11. Case-level bias and confidence
   12. Overall diagnosis

Scientific limitation:
    Pseudo-masks/development labels are not clinical expert-ground-truth.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part104_final_checkpoint_selection"
    / "checkpoints"
    / "final_segmentation_model.pth"
)

PART113_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part113_robust_loader_contract_and_full_inference"
)

COHORT_CSV = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part99_clinical_oriented_finetuning"
    / "part99_validation_cohort_reconstructed.csv"
)

PART113_CSV = PART113_DIR / "part113_validation_case_inference.csv"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part115_logit_probability_class_bias_diagnostic"
)

REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CASE_CSV = OUTPUT_DIR / "part115_case_logit_probability_diagnostic.csv"
CLASS_CSV = OUTPUT_DIR / "part115_class_probability_summary.csv"
MARGIN_CSV = OUTPUT_DIR / "part115_margin_entropy_summary.csv"
AGREE_CSV = OUTPUT_DIR / "part115_part113_prediction_agreement.csv"
SUMMARY_JSON = REPORT_DIR / "part115_logit_probability_class_bias_summary.json"
REPORT_TXT = REPORT_DIR / "part115_logit_probability_class_bias_report.txt"

NUM_CLASSES = 6
EXPECTED_SHAPE = (64, 96, 96)
VOXELS_PER_CASE = int(np.prod(EXPECTED_SHAPE))
EPS = 1e-8


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "prediction" not in data:
            raise RuntimeError(
                f"Part113 prediction file has no 'prediction': {path}"
            )
        pred = np.asarray(data["prediction"])

    if pred.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Part113 prediction shape mismatch: {pred.shape} "
            f"!= {EXPECTED_SHAPE}"
        )

    return pred.astype(np.int16, copy=False)


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=1)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main() -> int:
    banner("PART 115 — LOGIT / PROBABILITY / CLASS-BIAS DIAGNOSTIC")

    print(f"Project root : {ROOT}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Part113 CSV  : {PART113_CSV}")
    print(f"Cohort CSV   : {COHORT_CSV}")

    banner("1. INPUT VALIDATION")

    required = [
        CHECKPOINT,
        PART113_CSV,
        COHORT_CSV,
        ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py",
        ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py",
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        print("Missing required artifact(s):")
        for p in missing:
            print(f"  - {p}")
        print("\nFINAL STATUS: FAIL — REQUIRED_ARTIFACT_MISSING")
        return 1

    checkpoint_sha = sha256_file(CHECKPOINT)
    print(f"Final checkpoint SHA256 : {checkpoint_sha}")

    expected_sha = (
        "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
    )

    if checkpoint_sha != expected_sha:
        print("WARNING: checkpoint SHA differs from Part104 canonical SHA.")
        print(f"Expected: {expected_sha}")
        print(f"Actual  : {checkpoint_sha}")
    else:
        print("Checkpoint identity: PASS — matches Part104 selected checkpoint.")

    part113_df = pd.read_csv(PART113_CSV)
    cohort_df = pd.read_csv(COHORT_CSV)

    print(f"Part113 rows : {len(part113_df)}")
    print(f"Cohort rows  : {len(cohort_df)}")

    usable_df = part113_df[
        part113_df["status"].astype(str).str.upper() == "PASS"
    ].copy()

    if len(usable_df) == 0:
        print("\nFINAL STATUS: FAIL — NO_PART113_SUCCESSFUL_CASES")
        return 1

    # Exact study + series lookup from the canonical cohort.
    cohort_lookup: Dict[Tuple[str, str], pd.Series] = {}
    for _, row in cohort_df.iterrows():
        key = (str(row["study_id"]), str(row["series_id"]))
        if key not in cohort_lookup:
            cohort_lookup[key] = row

    banner("2. ENVIRONMENT + CANONICAL MODEL")

    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Inference device: {device}")

    # Exact architecture established in Parts 109–113.
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=12,
        spatial_dims=3,
        use_checkpoint=False,
    )

    runtime_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Canonical parameters: {runtime_params:,}")

    checkpoint_obj = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint_obj, dict):
        print("\nFINAL STATUS: FAIL — CHECKPOINT_NOT_DICT")
        return 1

    state = checkpoint_obj.get("model_state_dict")

    if not isinstance(state, dict):
        print("\nFINAL STATUS: FAIL — MODEL_STATE_DICT_MISSING")
        return 1

    try:
        model.load_state_dict(state, strict=True)
        print("Strict checkpoint load: PASS")
    except Exception as exc:
        print(f"Strict checkpoint load: FAIL — {exc}")
        print("\nFINAL STATUS: FAIL — CHECKPOINT_ARCHITECTURE_MISMATCH")
        return 1

    model.to(device)
    model.eval()

    banner("3. ESTABLISHED DATA LOADER")

    part9 = import_module_from_path(
        "part115_part9",
        ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py",
    )

    part11 = import_module_from_path(
        "part115_part11",
        ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py",
    )

    print("Part9 loader : FOUND")
    print("Part11 loader: FOUND")
    print(
        "Loader contract: Part113 established "
        "load_tensor_case(row, part9) -> (image, mask, info)"
    )

    banner("4. FRESH LOGIT / PROBABILITY INFERENCE")

    rows: List[Dict[str, Any]] = []
    class_prob_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_logit_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_logit_sq_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_argmax_count = np.zeros(NUM_CLASSES, dtype=np.int64)

    all_entropy = []
    all_confidence = []
    all_margin = []

    agreement_counts = {
        "same_argmax_voxels": 0,
        "total_voxels": 0,
    }

    failed = []

    start_all = time.time()

    for i, (_, result_row) in enumerate(usable_df.iterrows(), start=1):
        key = (
            str(result_row["study_id"]),
            str(result_row["series_id"]),
        )

        if key not in cohort_lookup:
            failed.append(
                f"{key}: not found in canonical cohort"
            )
            continue

        canonical_row = cohort_lookup[key]

        try:
            loaded = part11.load_tensor_case(
                canonical_row,
                part9,
            )

            if not isinstance(loaded, (tuple, list)) or len(loaded) < 2:
                raise RuntimeError(
                    "Unexpected load_tensor_case return."
                )

            image = loaded[0]
            target = loaded[1]

            if hasattr(image, "detach"):
                image_np = image.detach().cpu().numpy()
            else:
                image_np = np.asarray(image)

            if hasattr(target, "detach"):
                target_np = target.detach().cpu().numpy()
            else:
                target_np = np.asarray(target)

            if image_np.ndim == 3:
                image_np = image_np[None, None, ...]
            elif image_np.ndim == 4:
                image_np = image_np[None, ...]
            elif image_np.ndim != 5:
                raise RuntimeError(
                    f"Unexpected image dimensions: {image_np.shape}"
                )

            target_np = np.squeeze(target_np)

            if tuple(image_np.shape[-3:]) != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"Image shape {image_np.shape} does not end in "
                    f"{EXPECTED_SHAPE}"
                )

            if tuple(target_np.shape) != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"Target shape {target_np.shape} != {EXPECTED_SHAPE}"
                )

            image_tensor = torch.as_tensor(
                image_np,
                dtype=torch.float32,
                device=device,
            )

            with torch.inference_mode():
                logits = model(image_tensor)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                if logits.ndim != 5:
                    raise RuntimeError(
                        f"Unexpected model output shape: {tuple(logits.shape)}"
                    )

                if tuple(logits.shape[-3:]) != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"Output shape {tuple(logits.shape)} does not end in "
                        f"{EXPECTED_SHAPE}"
                    )

                probs = torch.softmax(logits.float(), dim=1)
                pred = torch.argmax(probs, dim=1)

            # Remove batch dimension.
            logits_v = logits[0]
            probs_v = probs[0]
            pred_v = pred[0]

            target_t = torch.as_tensor(
                target_np,
                dtype=torch.long,
                device=device,
            )

            entropy = entropy_from_probs(probs).flatten()
            confidence = probs.max(dim=1).values.flatten()

            bg_prob = probs_v[0].flatten()
            fg_prob = probs_v[1:].sum(dim=0).flatten()

            # Positive margin means foreground probability exceeds background.
            fg_bg_margin = fg_prob - bg_prob

            # Class-1 vs all-other class probability margin.
            class1_margin = (
                probs_v[1].flatten()
                - probs_v[2:].sum(dim=0).flatten()
            )

            logits_cpu = logits_v.detach().cpu()
            probs_cpu = probs_v.detach().cpu()
            pred_cpu = pred_v.detach().cpu()

            pred_np = pred_cpu.numpy().astype(np.int16)

            # Part113 prediction file agreement.
            pred_rel = str(
                result_row.get("prediction_file", "")
            ).strip()

            if pred_rel:
                pred_path = ROOT / pred_rel
                if not pred_path.exists():
                    pred_path = PART113_DIR / "predictions" / Path(
                        pred_rel
                    ).name
            else:
                pred_path = None

            if pred_path is not None and pred_path.exists():
                old_pred = load_prediction(pred_path)
                same = int(np.sum(old_pred == pred_np))
                total = int(old_pred.size)

                agreement_counts["same_argmax_voxels"] += same
                agreement_counts["total_voxels"] += total

                agreement_pct = same / total
            else:
                agreement_pct = float("nan")

            # Per-case statistics.
            case = {
                "case_index": i,
                "row_position": result_row.get("row_position"),
                "study_id": result_row.get("study_id"),
                "series_id": result_row.get("series_id"),
                "series_description": result_row.get(
                    "series_description",
                    "",
                ),
                "mean_entropy": float(entropy.mean().item()),
                "mean_confidence": float(confidence.mean().item()),
                "p95_confidence": float(
                    torch.quantile(confidence, 0.95).item()
                ),
                "mean_fg_probability": float(
                    fg_prob.mean().item()
                ),
                "mean_background_probability": float(
                    bg_prob.mean().item()
                ),
                "mean_fg_minus_bg_margin": float(
                    fg_bg_margin.mean().item()
                ),
                "positive_fg_bg_margin_fraction": float(
                    (fg_bg_margin > 0).float().mean().item()
                ),
                "mean_class1_probability": float(
                    probs_v[1].mean().item()
                ),
                "mean_class1_minus_other_fg_probability": float(
                    class1_margin.mean().item()
                ),
                "class1_margin_positive_fraction": float(
                    (class1_margin > 0).float().mean().item()
                ),
                "part113_argmax_agreement_fraction": agreement_pct,
            }

            for c in range(NUM_CLASSES):
                ls = tensor_stats(logits_v[c])
                ps = tensor_stats(probs_v[c])

                case[f"class_{c}_logit_mean"] = ls["mean"]
                case[f"class_{c}_logit_std"] = ls["std"]
                case[f"class_{c}_logit_min"] = ls["min"]
                case[f"class_{c}_logit_max"] = ls["max"]
                case[f"class_{c}_prob_mean"] = ps["mean"]
                case[f"class_{c}_prob_std"] = ps["std"]
                case[f"class_{c}_prob_min"] = ps["min"]
                case[f"class_{c}_prob_max"] = ps["max"]

                pred_count = int(
                    (pred_v == c).sum().item()
                )
                target_count = int(
                    (target_t == c).sum().item()
                )

                case[f"class_{c}_pred_voxels"] = pred_count
                case[f"class_{c}_target_voxels"] = target_count
                case[f"class_{c}_pred_fraction"] = (
                    pred_count / VOXELS_PER_CASE
                )

                class_prob_sum[c] += float(
                    probs_v[c].sum().item()
                )

                class_logit_sum[c] += float(
                    logits_v[c].sum().item()
                )

                class_logit_sq_sum[c] += float(
                    (logits_v[c].float() ** 2).sum().item()
                )

                class_argmax_count[c] += pred_count

            # Target-conditioned probability mass.
            for c in range(NUM_CLASSES):
                target_mask = target_t == c
                if target_mask.any():
                    case[
                        f"class_{c}_prob_on_target_class"
                    ] = float(
                        probs_v[c][target_mask].mean().item()
                    )
                    case[
                        f"class_{c}_argmax_accuracy_on_target_class"
                    ] = float(
                        (pred_v[target_mask] == c)
                        .float()
                        .mean()
                        .item()
                    )
                else:
                    case[
                        f"class_{c}_prob_on_target_class"
                    ] = float("nan")
                    case[
                        f"class_{c}_argmax_accuracy_on_target_class"
                    ] = float("nan")

            rows.append(case)

            all_entropy.append(float(entropy.mean().item()))
            all_confidence.append(float(confidence.mean().item()))
            all_margin.append(float(fg_bg_margin.mean().item()))

            if i == 1 or i % 10 == 0:
                print(
                    f"[{i:03d}/{len(usable_df):03d}] "
                    f"study={result_row.get('study_id')} "
                    f"series={result_row.get('series_id')} "
                    f"mean_fg_p={case['mean_fg_probability']:.4f} "
                    f"mean_bg_p={case['mean_background_probability']:.4f} "
                    f"mean_conf={case['mean_confidence']:.4f} "
                    f"entropy={case['mean_entropy']:.4f}"
                )

        except Exception as exc:
            failed.append(
                f"row={result_row.get('row_position')}, "
                f"study={result_row.get('study_id')}, "
                f"series={result_row.get('series_id')}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"[{i:03d}/{len(usable_df):03d}] FAIL: "
                f"{type(exc).__name__}: {exc}"
            )

    elapsed = time.time() - start_all

    if not rows:
        print("\nFINAL STATUS: FAIL — NO_CASES_COMPLETED")
        return 1

    case_df = pd.DataFrame(rows)
    case_df.to_csv(CASE_CSV, index=False)

    banner("5. AGGREGATE CLASS PROBABILITY SUMMARY")

    n_cases = len(case_df)

    # Each case has exactly VOXELS_PER_CASE voxels.
    total_voxels = n_cases * VOXELS_PER_CASE

    class_summary = []

    print(
        f"{'Class':<8}"
        f"{'Mean P':>12}"
        f"{'Argmax %':>14}"
        f"{'Mean logit':>16}"
        f"{'Logit SD':>14}"
        f"{'Pred voxels':>16}"
    )

    for c in range(NUM_CLASSES):
        mean_prob = (
            class_prob_sum[c] / total_voxels
        )

        pred_fraction = (
            class_argmax_count[c] / total_voxels
        )

        mean_logit = (
            class_logit_sum[c] / total_voxels
        )

        variance = (
            class_logit_sq_sum[c] / total_voxels
            - mean_logit ** 2
        )

        logit_std = math.sqrt(max(variance, 0.0))

        print(
            f"{c:<8}"
            f"{mean_prob:>12.6f}"
            f"{pred_fraction * 100:>13.3f}%"
            f"{mean_logit:>16.6f}"
            f"{logit_std:>14.6f}"
            f"{int(class_argmax_count[c]):>16,}"
        )

        class_summary.append(
            {
                "class_id": c,
                "mean_probability": mean_prob,
                "argmax_fraction": pred_fraction,
                "mean_logit": mean_logit,
                "logit_std": logit_std,
                "predicted_voxels": int(
                    class_argmax_count[c]
                ),
            }
        )

    class_summary_df = pd.DataFrame(class_summary)
    class_summary_df.to_csv(CLASS_CSV, index=False)

    banner("6. CONFIDENCE / ENTROPY / MARGIN SUMMARY")

    mean_entropy = float(np.mean(all_entropy))
    mean_confidence = float(np.mean(all_confidence))
    mean_fg_margin = float(np.mean(all_margin))

    mean_fg_probability = float(
        case_df["mean_fg_probability"].mean()
    )

    mean_bg_probability = float(
        case_df["mean_background_probability"].mean()
    )

    positive_fg_margin_fraction = float(
        case_df["positive_fg_bg_margin_fraction"].mean()
    )

    mean_class1_probability = float(
        case_df["mean_class1_probability"].mean()
    )

    class1_positive_margin_fraction = float(
        case_df["class1_margin_positive_fraction"].mean()
    )

    print(f"Mean voxel entropy                 : {mean_entropy:.6f}")
    print(f"Mean max-class confidence          : {mean_confidence:.6f}")
    print(f"Mean foreground probability       : {mean_fg_probability:.6f}")
    print(f"Mean background probability       : {mean_bg_probability:.6f}")
    print(f"Mean FG - BG probability margin   : {mean_fg_margin:.6f}")
    print(
        f"Mean fraction with FG > BG        : "
        f"{positive_fg_margin_fraction * 100:.3f}%"
    )
    print(
        f"Mean class-1 probability          : "
        f"{mean_class1_probability:.6f}"
    )
    print(
        f"Class-1 > other-FG margin fraction: "
        f"{class1_positive_margin_fraction * 100:.3f}%"
    )

    margin_df = pd.DataFrame(
        [
            {
                "metric": "mean_entropy",
                "value": mean_entropy,
            },
            {
                "metric": "mean_confidence",
                "value": mean_confidence,
            },
            {
                "metric": "mean_foreground_probability",
                "value": mean_fg_probability,
            },
            {
                "metric": "mean_background_probability",
                "value": mean_bg_probability,
            },
            {
                "metric": "mean_fg_minus_bg_margin",
                "value": mean_fg_margin,
            },
            {
                "metric": "fraction_fg_probability_gt_background",
                "value": positive_fg_margin_fraction,
            },
            {
                "metric": "mean_class1_probability",
                "value": mean_class1_probability,
            },
            {
                "metric": "fraction_class1_gt_other_foreground",
                "value": class1_positive_margin_fraction,
            },
        ]
    )
    margin_df.to_csv(MARGIN_CSV, index=False)

    banner("7. PART113 ARGMAX REPRODUCTION")

    agreement_fraction = (
        agreement_counts["same_argmax_voxels"]
        / max(agreement_counts["total_voxels"], 1)
    )

    print(
        f"Same argmax voxels : "
        f"{agreement_counts['same_argmax_voxels']:,}"
    )
    print(
        f"Total voxels       : "
        f"{agreement_counts['total_voxels']:,}"
    )
    print(
        f"Agreement           : "
        f"{agreement_fraction * 100:.6f}%"
    )

    pd.DataFrame(
        [
            {
                "same_argmax_voxels": agreement_counts[
                    "same_argmax_voxels"
                ],
                "total_voxels": agreement_counts[
                    "total_voxels"
                ],
                "agreement_fraction": agreement_fraction,
            }
        ]
    ).to_csv(AGREE_CSV, index=False)

    banner("8. TARGET-CONDITIONED CLASS DIAGNOSTIC")

    for c in range(NUM_CLASSES):
        prob_col = f"class_{c}_prob_on_target_class"
        acc_col = f"class_{c}_argmax_accuracy_on_target_class"

        if prob_col in case_df.columns:
            p = case_df[prob_col].to_numpy(dtype=float)
            a = case_df[acc_col].to_numpy(dtype=float)

            p = p[np.isfinite(p)]
            a = a[np.isfinite(a)]

            if p.size:
                mean_p = float(np.mean(p))
                median_p = float(np.median(p))
            else:
                mean_p = float("nan")
                median_p = float("nan")

            if a.size:
                mean_a = float(np.mean(a))
            else:
                mean_a = float("nan")

            print(
                f"Target class {c}: "
                f"mean P(class={c}|target={c})="
                f"{mean_p:.6f}, "
                f"median={median_p:.6f}, "
                f"argmax accuracy={mean_a:.6f}"
            )

    banner("9. CLASS-BIAS DIAGNOSIS")

    argmax_fraction = (
        class_argmax_count / total_voxels
    )

    fg_argmax = argmax_fraction[1:]
    fg_total = fg_argmax.sum()

    dominant_fg_class = int(np.argmax(fg_argmax) + 1)
    dominant_fg_share = (
        fg_argmax[dominant_fg_class - 1] / fg_total
        if fg_total > 0
        else 0.0
    )

    bg_argmax_fraction = float(argmax_fraction[0])
    fg_argmax_fraction = float(fg_total)

    # Probability-mass diagnostics are more informative than argmax alone.
    class1_prob_share_of_fg = (
        mean_class1_probability
        / max(mean_fg_probability, EPS)
    )

    print(
        f"Background argmax fraction       : "
        f"{bg_argmax_fraction * 100:.3f}%"
    )
    print(
        f"Foreground argmax fraction       : "
        f"{fg_argmax_fraction * 100:.3f}%"
    )
    print(
        f"Dominant foreground class        : {dominant_fg_class}"
    )
    print(
        f"Dominant FG class share           : "
        f"{dominant_fg_share * 100:.3f}%"
    )
    print(
        f"Class-1 share of foreground P mass: "
        f"{class1_prob_share_of_fg * 100:.3f}%"
    )

    if (
        fg_argmax_fraction > 0.05
        and class1_prob_share_of_fg > 0.70
    ):
        diagnosis = "CLASS_1_FOREGROUND_BIAS"
    elif fg_argmax_fraction > 0.05:
        diagnosis = "GENERAL_FOREGROUND_BIAS"
    elif mean_confidence > 0.80 and fg_argmax_fraction < 0.05:
        diagnosis = "LOW_FOREGROUND_WITH_HIGH_CONFIDENCE"
    else:
        diagnosis = "NO_SINGLE_LOGIT_CLASS_BIAS"

    if mean_confidence > 0.85 and fg_argmax_fraction > 0.05:
        confidence_diagnosis = "HIGH_CONFIDENCE_WRONG_FOREGROUND_PATTERN"
    elif mean_confidence < 0.50:
        confidence_diagnosis = "LOW_CONFIDENCE_MODEL_OUTPUTS"
    else:
        confidence_diagnosis = "INTERMEDIATE_CONFIDENCE"

    print(f"Class-bias diagnosis              : {diagnosis}")
    print(f"Confidence diagnosis               : {confidence_diagnosis}")

    banner("10. FINAL DIAGNOSTIC DECISION")

    # These are diagnostic categories, not clinical performance claims.
    if (
        diagnosis == "CLASS_1_FOREGROUND_BIAS"
        and fg_argmax_fraction > 0.05
    ):
        primary_diagnosis = (
            "CLASS_1_LOGIT_PROBABILITY_DOMINANCE_WITH_FOREGROUND_OVERSEGMENTATION"
        )
    elif diagnosis == "GENERAL_FOREGROUND_BIAS":
        primary_diagnosis = (
            "GENERAL_FOREGROUND_LOGIT_BIAS_WITH_OVERSEGMENTATION"
        )
    elif confidence_diagnosis == "HIGH_CONFIDENCE_WRONG_FOREGROUND_PATTERN":
        primary_diagnosis = (
            "HIGH_CONFIDENCE_FOREGROUND_MISCLASSIFICATION"
        )
    else:
        primary_diagnosis = (
            "LOGIT_PATTERN_REQUIRES_DEEPER_ANALYSIS"
        )

    print(f"Primary diagnosis: {primary_diagnosis}")

    banner("11. OUTPUTS")

    print(f"Case diagnostic CSV : {CASE_CSV}")
    print(f"Class summary CSV   : {CLASS_CSV}")
    print(f"Margin summary CSV  : {MARGIN_CSV}")
    print(f"Agreement CSV       : {AGREE_CSV}")
    print(f"Summary JSON        : {SUMMARY_JSON}")
    print(f"Report TXT          : {REPORT_TXT}")
    print(f"Inference time      : {elapsed:.2f} sec")
    print(f"Completed cases     : {len(rows)}")
    print(f"Failed cases        : {len(failed)}")

    summary = {
        "part": 115,
        "status": (
            "PASS — LOGIT_PROBABILITY_CLASS_BIAS_DIAGNOSTIC_COMPLETED"
        ),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha,
        "expected_part104_sha256": expected_sha,
        "checkpoint_matches_part104": checkpoint_sha == expected_sha,
        "canonical_model": {
            "type": "MONAI SwinUNETR",
            "in_channels": 1,
            "out_channels": NUM_CLASSES,
            "feature_size": 12,
            "spatial_dims": 3,
            "use_checkpoint": False,
            "runtime_parameters": runtime_params,
        },
        "part113_rows": int(len(part113_df)),
        "completed_cases": int(len(rows)),
        "failed_cases": int(len(failed)),
        "failures": failed,
        "mean_entropy": mean_entropy,
        "mean_confidence": mean_confidence,
        "mean_foreground_probability": mean_fg_probability,
        "mean_background_probability": mean_bg_probability,
        "mean_fg_minus_bg_margin": mean_fg_margin,
        "fraction_fg_gt_background": positive_fg_margin_fraction,
        "mean_class1_probability": mean_class1_probability,
        "fraction_class1_gt_other_foreground": class1_positive_margin_fraction,
        "argmax_fraction_by_class": argmax_fraction.tolist(),
        "dominant_foreground_class": dominant_fg_class,
        "dominant_foreground_class_share": dominant_fg_share,
        "class1_probability_share_of_foreground": class1_prob_share_of_fg,
        "part113_argmax_agreement": agreement_fraction,
        "class_bias_diagnosis": diagnosis,
        "confidence_diagnosis": confidence_diagnosis,
        "primary_diagnosis": primary_diagnosis,
        "inference_seconds": elapsed,
        "scientific_note": (
            "Targets are project pseudo-masks/development labels and are not "
            "clinical expert-groundtruth."
        ),
        "next_step_note": (
            "Use this diagnostic to decide whether a targeted training/loss "
            "investigation is scientifically justified. Do not infer clinical "
            "performance from these pseudo-mask diagnostics."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "PART 115 — LOGIT / PROBABILITY / CLASS-BIAS DIAGNOSTIC",
        "",
        f"Status: {summary['status']}",
        f"Checkpoint SHA256: {checkpoint_sha}",
        f"Completed cases: {len(rows)}",
        f"Failed cases: {len(failed)}",
        "",
        f"Mean entropy: {mean_entropy:.6f}",
        f"Mean confidence: {mean_confidence:.6f}",
        f"Mean foreground probability: {mean_fg_probability:.6f}",
        f"Mean background probability: {mean_bg_probability:.6f}",
        f"Mean FG-BG margin: {mean_fg_margin:.6f}",
        f"Fraction FG > BG: {positive_fg_margin_fraction:.6f}",
        f"Mean class-1 probability: {mean_class1_probability:.6f}",
        f"Class-1 > other-FG fraction: {class1_positive_margin_fraction:.6f}",
        "",
        f"Dominant foreground class: {dominant_fg_class}",
        f"Dominant foreground share: {dominant_fg_share:.6f}",
        f"Class-1 probability share of foreground: {class1_prob_share_of_fg:.6f}",
        f"Part113 argmax agreement: {agreement_fraction:.6f}",
        "",
        f"Class-bias diagnosis: {diagnosis}",
        f"Confidence diagnosis: {confidence_diagnosis}",
        f"Primary diagnosis: {primary_diagnosis}",
        "",
        "Scientific limitation:",
        "The targets are project pseudo-masks/development labels.",
        "These diagnostics are not clinical expert-groundtruth performance.",
    ]

    REPORT_TXT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 115 FINAL RESULT")
    print(
        "PASS — LOGIT / PROBABILITY / CLASS-BIAS DIAGNOSTIC COMPLETED"
    )
    print(f"Primary diagnosis: {primary_diagnosis}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
