"""
==============================================================================
PHASE 4 - PART 37
RSNA-ONLY PART 15 METRIC-PATH / CHECKPOINT-ORIGIN FORENSIC AUDIT
==============================================================================

Purpose
-------
Part 35 proved that the saved Part 15 checkpoint predicts background for all
100 reconstructed validation cases.

Part 36 proved that the current corrected Part 11 Dice API is incompatible
with the historical Part 15 validation call.

Part 37 performs a deeper forensic audit without training:

    A. Inspect the complete Part 15 source for metric/checkpoint logic.
    B. Inspect all available Part 15 checkpoint artifacts.
    C. Inspect Part 15 history and checkpoint metadata.
    D. Verify whether multiple checkpoint files exist.
    E. Evaluate the available Part 15 checkpoint against the exact 100-case
       validation cohort.
    F. Determine whether the stored 0.668 value is traceable to the current
       checkpoint.
    G. Check whether the source contains a fallback / surrogate metric,
       target Dice implementation, or checkpoint-selection anomaly.
    H. Produce a machine-readable forensic report.

NO TRAINING.
NO OPTIMIZER.
NO WEIGHT MODIFICATION.
NO SPIDER.
NO RSNA TEST SET.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
import re
import sys
import traceback
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART8_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part8_dataset_construction"
)

VAL_MANIFEST = (
    PART8_DIR
    / "manifests"
    / "rsna_part8_validation_manifest.csv"
)

PART11_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_SOURCE = (
    SRC_DIR
    / "segmentation_rsna_part15_extended_controlled_training.py"
)

PART15_CORRECTED = (
    SRC_DIR
    / "segmentation_rsna_part15_extended_controlled_training_corrected.py"
)

PART15_CORRECTED_V2 = (
    SRC_DIR
    / "segmentation_rsna_part15_extended_controlled_training_corrected_v2.py"
)

PART15_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT_DIR = PART15_OUTPUT / "checkpoints"

PART15_CHECKPOINT = (
    PART15_CHECKPOINT_DIR / "best_model.pth"
)

PART15_HISTORY = (
    PART15_OUTPUT / "part15_training_history.csv"
)

PART15_TRAIN_COHORT = (
    PART15_OUTPUT / "part15_train_cohort.csv"
)

PART15_VAL_COHORT = (
    PART15_OUTPUT / "part15_validation_cohort.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part37_checkpoint_metric_forensic_audit"
)

CHECKPOINT_CSV = OUTPUT_DIR / "part37_checkpoint_inventory.csv"
SOURCE_CSV = OUTPUT_DIR / "part37_source_metric_findings.csv"
CASE_CSV = OUTPUT_DIR / "part37_case_checkpoint_reconstruction.csv"
CLASS_CSV = OUTPUT_DIR / "part37_class_metric_summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase4_part37_summary.json"
REPORT_TXT = OUTPUT_DIR / "phase4_part37_report.txt"


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

SEED = 42
VAL_CASES = 100
NUM_CLASSES = 6
PATCH_SIZE = (64, 96, 96)
FOREGROUND = [1, 2, 3, 4, 5]

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def section(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def import_module(path: Path, name: str):
    spec = spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def norm_id(value):
    if pd.isna(value):
        return ""

    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def row_ids(row):
    return norm_id(row.get("study_id")), norm_id(row.get("series_id"))


def sha256_file(path: Path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


# =============================================================================
# SOURCE FORENSICS
# =============================================================================

def read_source(path: Path):
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def find_source_patterns(text: str):
    patterns = {
        "dice_from_prediction": r"dice_from_prediction",
        "val_dice": r"\bval_dice\b",
        "train_dice": r"\btrain_dice\b",
        "best_val_dice": r"\bbest_val_dice\b",
        "save_checkpoint": r"save_checkpoint",
        "torch\.save": r"torch\.save",
        "load_checkpoint": r"load_checkpoint",
        "argmax": r"argmax",
        "class_dice\.get": r"class_dice\s*\.\s*get",
        "NUM_CLASSES": r"NUM_CLASSES",
        "optimizer": r"\boptimizer\b",
        "model\.eval": r"model\.eval",
        "model\.train": r"model\.train",
        "softmax": r"\bsoftmax\b",
        "one_hot": r"one_hot",
        "background": r"background",
        "foreground": r"foreground",
    }

    findings = []

    lines = text.splitlines()

    for label, pattern in patterns.items():
        regex = re.compile(pattern, re.IGNORECASE)

        matches = [
            i + 1
            for i, line in enumerate(lines)
            if regex.search(line)
        ]

        findings.append({
            "pattern": label,
            "occurrences": len(matches),
            "line_numbers": ",".join(map(str, matches[:100])),
        })

    return findings


def extract_functions(path: Path):
    text = read_source(path)

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "syntax_error": repr(exc),
            "functions": {},
        }

    lines = text.splitlines()
    functions = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = "\n".join(
                lines[node.lineno - 1:node.end_lineno]
            )

    return {
        "syntax_error": None,
        "functions": functions,
    }


def extract_metric_calls(source: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = None

        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name != "dice_from_prediction":
            continue

        results.append({
            "line": node.lineno,
            "positional_args": len(node.args),
            "keyword_args": [
                kw.arg for kw in node.keywords
            ],
        })

    return results


def extract_checkpoint_save_calls(source: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = None

        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name in {
            "save_checkpoint",
            "torch.save",
        }:
            results.append({
                "line": node.lineno,
                "function": name,
                "positional_args": len(node.args),
                "keyword_args": [
                    kw.arg for kw in node.keywords
                ],
            })

    return results


# =============================================================================
# CHECKPOINT INVENTORY
# =============================================================================

def inventory_checkpoints():
    records = []

    if not PART15_CHECKPOINT_DIR.exists():
        return records

    for path in sorted(
        PART15_CHECKPOINT_DIR.rglob("*")
    ):
        if not path.is_file():
            continue

        try:
            stat = path.stat()

            records.append({
                "filename": path.name,
                "relative_path": str(
                    path.relative_to(PROJECT_ROOT)
                ),
                "suffix": path.suffix,
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "sha256": sha256_file(path),
            })

        except Exception as exc:
            records.append({
                "filename": path.name,
                "relative_path": str(
                    path.relative_to(PROJECT_ROOT)
                ),
                "error": repr(exc),
            })

    return records


def inspect_checkpoint(path: Path, device):
    result = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return result

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        result["python_type"] = str(
            type(checkpoint)
        )

        if isinstance(checkpoint, dict):
            result["keys"] = sorted(
                str(k)
                for k in checkpoint.keys()
            )

            for key in (
                "epoch",
                "best_val_dice",
                "val_dice",
                "train_dice",
                "best_dice",
            ):
                if key in checkpoint:
                    value = checkpoint[key]

                    if isinstance(
                        value,
                        (int, float, str, bool)
                    ) or value is None:
                        result[key] = value
                    else:
                        result[key] = str(
                            type(value)
                        )

            for key in (
                "model_state_dict",
                "state_dict",
                "model",
            ):
                if key in checkpoint:
                    value = checkpoint[key]

                    if isinstance(value, dict):
                        result[
                            f"{key}_parameter_count"
                        ] = int(
                            sum(
                                v.numel()
                                for v in value.values()
                                if isinstance(
                                    v,
                                    torch.Tensor
                                )
                            )
                        )
                        result[
                            f"{key}_tensor_count"
                        ] = int(
                            sum(
                                1
                                for v in value.values()
                                if isinstance(
                                    v,
                                    torch.Tensor
                                )
                            )
                        )

        return result

    except Exception as exc:
        result["load_error"] = repr(exc)
        return result


# =============================================================================
# MODEL / DATA
# =============================================================================

def load_model(part11, device):
    model = part11.create_model(device)
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(
        PART15_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    state = None

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            value = checkpoint.get(key)

            if isinstance(value, dict):
                state = value
                break

    if state is None:
        if (
            isinstance(checkpoint, dict)
            and all(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            )
        ):
            state = checkpoint

    if state is None:
        raise RuntimeError(
            "Could not locate model state in Part 15 checkpoint."
        )

    cleaned = {
        (
            k[7:]
            if k.startswith("module.")
            else k
        ): v
        for k, v in state.items()
    }

    load_result = model.load_state_dict(
        cleaned,
        strict=False,
    )

    return (
        model,
        checkpoint,
        load_result,
    )


def ensure_image(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = torch.as_tensor(x).float()

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4 and x.shape[0] == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 5:
        raise ValueError(
            f"Unexpected image shape {tuple(x.shape)}"
        )

    return x.contiguous()


def ensure_mask(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = torch.as_tensor(x).long()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.ndim != 3:
        raise ValueError(
            f"Unexpected mask shape {tuple(x.shape)}"
        )

    return x.contiguous()


def normalize_case(image, mask):
    image = ensure_image(image)
    mask = ensure_mask(mask)

    if tuple(image.shape[-3:]) != PATCH_SIZE:
        image = F.interpolate(
            image,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

    if tuple(mask.shape) != PATCH_SIZE:
        mask = F.interpolate(
            mask[None, None].float(),
            size=PATCH_SIZE,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

    return image, mask


def load_case(part11, part9, row):
    image, mask, info = part11.load_tensor_case(
        row,
        part9,
    )

    if isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]

        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]

        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    processed = part11.preprocess_case(
        image,
        mask,
    )

    if isinstance(processed, tuple):
        image = processed[0]
        mask = processed[1]
    else:
        image = processed

    return normalize_case(
        image,
        mask,
    )


# =============================================================================
# METRIC
# =============================================================================

def metric_from_logits(
    logits,
    target,
):
    prediction = torch.argmax(
        logits,
        dim=1,
    ).squeeze(0)

    values = {}

    for c in FOREGROUND:
        p = prediction == c
        t = target == c

        tp = int((p & t).sum().item())
        fp = int((p & ~t).sum().item())
        fn = int((~p & t).sum().item())

        denominator = (
            2 * tp +
            fp +
            fn
        )

        values[c] = (
            float("nan")
            if denominator == 0
            else 2.0 * tp / denominator
        )

    valid = [
        x
        for x in values.values()
        if not math.isnan(x)
    ]

    mean = (
        float(np.mean(valid))
        if valid
        else float("nan")
    )

    return (
        prediction,
        mean,
        values,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    section("PHASE 4 - PART 37")
    print(
        "RSNA-ONLY PART 15 METRIC-PATH / CHECKPOINT-ORIGIN "
        "FORENSIC AUDIT"
    )
    print()
    print("Evaluation / source audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("No optimizer step is performed.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    section("PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "Part 8 validation manifest": VAL_MANIFEST,
        "Part 11 source": PART11_SOURCE,
        "Part 15 source": PART15_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 15 validation cohort": PART15_VAL_COHORT,
    }

    for name, path in paths.items():
        print(
            f"{name:<36}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )

    required = [
        RSNA_ROOT,
        VAL_MANIFEST,
        PART11_SOURCE,
        PART15_SOURCE,
        PART15_CHECKPOINT,
        PART15_HISTORY,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n"
            + "\n".join(missing)
        )

    # -------------------------------------------------------------------------
    # SOURCE AUDIT
    # -------------------------------------------------------------------------
    section("PART 15 SOURCE FORENSICS")

    part15_text = read_source(
        PART15_SOURCE
    )

    findings = find_source_patterns(
        part15_text
    )

    source_df = pd.DataFrame(
        findings
    )

    print(
        source_df.to_string(
            index=False
        )
    )

    metric_calls = extract_metric_calls(
        part15_text
    )

    save_calls = extract_checkpoint_save_calls(
        part15_text
    )

    print()
    print(
        "dice_from_prediction() calls found : "
        f"{len(metric_calls)}"
    )

    for item in metric_calls:
        print(
            f"  line {item['line']}: "
            f"{item['positional_args']} positional args"
        )

    print(
        "Checkpoint-save calls found         : "
        f"{len(save_calls)}"
    )

    for item in save_calls:
        print(
            f"  line {item['line']}: "
            f"{item['function']}()"
        )

    # -------------------------------------------------------------------------
    # OTHER PART 15 SOURCES
    # -------------------------------------------------------------------------
    section("PART 15 CORRECTED SOURCE INVENTORY")

    for path in (
        PART15_CORRECTED,
        PART15_CORRECTED_V2,
    ):
        if path.exists():
            text = read_source(path)
            calls = extract_metric_calls(text)

            print()
            print(f"FILE: {path.name}")
            print(
                f"  Size                           : "
                f"{path.stat().st_size:,} bytes"
            )
            print(
                f"  Dice API calls                 : "
                f"{len(calls)}"
            )

            for call in calls:
                print(
                    f"    line {call['line']} -> "
                    f"{call['positional_args']} positional args"
                )
        else:
            print(
                f"{path.name:<60}: NOT FOUND"
            )

    # -------------------------------------------------------------------------
    # CHECKPOINT INVENTORY
    # -------------------------------------------------------------------------
    section("PART 15 CHECKPOINT INVENTORY")

    checkpoint_inventory = inventory_checkpoints()

    if not checkpoint_inventory:
        print("No checkpoint files found.")
    else:
        checkpoint_df = pd.DataFrame(
            checkpoint_inventory
        )

        display_columns = [
            "filename",
            "size_bytes",
            "sha256",
        ]

        print(
            checkpoint_df[
                [
                    c
                    for c in display_columns
                    if c in checkpoint_df.columns
                ]
            ].to_string(
                index=False
            )
        )

        checkpoint_df.to_csv(
            CHECKPOINT_CSV,
            index=False,
        )

    # -------------------------------------------------------------------------
    # CHECKPOINT METADATA
    # -------------------------------------------------------------------------
    section("BEST CHECKPOINT METADATA")

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_info = inspect_checkpoint(
        PART15_CHECKPOINT,
        device,
    )

    for key, value in checkpoint_info.items():
        if key == "keys":
            print(
                f"{key:<36}: "
                f"{', '.join(value)}"
            )
        else:
            print(
                f"{key:<36}: {value}"
            )

    # -------------------------------------------------------------------------
    # HISTORY
    # -------------------------------------------------------------------------
    section("PART 15 HISTORY FORENSICS")

    history = pd.read_csv(
        PART15_HISTORY
    )

    print(
        f"History rows                         : "
        f"{len(history)}"
    )

    if "val_dice" in history.columns:
        best_idx = history["val_dice"].idxmax()
        best_row = history.loc[best_idx]

        print()
        print(
            "Maximum recorded validation Dice"
        )
        print(
            best_row.to_string()
        )

        print()
        print(
            "All validation Dice values"
        )

        for _, row in history.iterrows():
            print(
                f"  epoch={row.get('epoch')} "
                f"val_dice={row.get('val_dice')}"
            )
    else:
        best_idx = None
        best_row = None

    # -------------------------------------------------------------------------
    # EXACT COHORT
    # -------------------------------------------------------------------------
    section("EXACT VALIDATION COHORT")

    part11 = import_module(
        PART11_SOURCE,
        "part11_for_part37",
    )

    val_manifest = pd.read_csv(
        VAL_MANIFEST
    )

    cohort = part11.select_pilot_rows(
        VAL_MANIFEST,
        VAL_CASES,
        SEED,
    ).reset_index(drop=True)

    print(
        f"Validation manifest rows             : "
        f"{len(val_manifest)}"
    )

    print(
        f"Reconstructed cohort rows            : "
        f"{len(cohort)}"
    )

    if PART15_VAL_COHORT.exists():
        saved_cohort = pd.read_csv(
            PART15_VAL_COHORT
        )

        reconstructed_pairs = [
            (
                str(x),
                str(y),
            )
            for x, y in zip(
                cohort["study_id"],
                cohort["series_id"],
            )
        ]

        saved_pairs = [
            (
                str(x),
                str(y),
            )
            for x, y in zip(
                saved_cohort["study_id"],
                saved_cohort["series_id"],
            )
        ]

        exact = (
            reconstructed_pairs ==
            saved_pairs
        )

        print(
            f"Saved Part 15 cohort rows             : "
            f"{len(saved_cohort)}"
        )

        print(
            f"Exact row-order match                 : "
            f"{exact}"
        )

    if len(cohort) != VAL_CASES:
        raise RuntimeError(
            f"Expected {VAL_CASES} cases; "
            f"got {len(cohort)}."
        )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    section("LOADING SAVED BEST CHECKPOINT")

    model, checkpoint, load_result = load_model(
        part11,
        device,
    )

    print(
        f"Total parameters                     : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        f"Missing keys                         : "
        f"{len(load_result.missing_keys)}"
    )

    print(
        f"Unexpected keys                      : "
        f"{len(load_result.unexpected_keys)}"
    )

    if isinstance(checkpoint, dict):
        print(
            f"Checkpoint epoch                    : "
            f"{checkpoint.get('epoch', 'N/A')}"
        )

        print(
            f"Checkpoint best Dice               : "
            f"{checkpoint.get('best_val_dice', 'N/A')}"
        )

    # -------------------------------------------------------------------------
    # PART 9
    # -------------------------------------------------------------------------
    section("LOADING PART 9 THROUGH PART 11")

    part9 = part11.load_part9_module()

    print(
        "✓ Part 9 loaded through corrected Part 11."
    )

    # -------------------------------------------------------------------------
    # CASE EVALUATION
    # -------------------------------------------------------------------------
    section("RUNNING CHECKPOINT FORENSIC EVALUATION")

    case_records = []

    aggregate = {
        c: {
            "target": 0,
            "prediction": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for c in FOREGROUND
    }

    failed = []

    with torch.no_grad():
        for index, row in cohort.iterrows():

            case_index = index + 1
            study, series = row_ids(row)

            try:
                image, target = load_case(
                    part11,
                    part9,
                    row,
                )

                image = image.to(
                    device
                )

                target = target.to(
                    device
                )

                logits = model(
                    image
                )

                prediction, mean_dice, class_dice = (
                    metric_from_logits(
                        logits,
                        target,
                    )
                )

                prediction_fg = int(
                    sum(
                        int(
                            (
                                prediction == c
                            ).sum().item()
                        )
                        for c in FOREGROUND
                    )
                )

                target_fg = int(
                    sum(
                        int(
                            (
                                target == c
                            ).sum().item()
                        )
                        for c in FOREGROUND
                    )
                )

                record = {
                    "case_index": case_index,
                    "study_id": study,
                    "series_id": series,
                    "status": "OK",
                    "target_foreground_voxels": target_fg,
                    "prediction_foreground_voxels": prediction_fg,
                    "prediction_target_foreground_ratio": (
                        prediction_fg / target_fg
                        if target_fg
                        else float("nan")
                    ),
                    "mean_foreground_dice": mean_dice,
                    "unique_prediction_labels": ",".join(
                        map(
                            str,
                            sorted(
                                int(x)
                                for x in torch.unique(
                                    prediction
                                ).cpu()
                            ),
                        )
                    ),
                }

                for c in FOREGROUND:
                    p = prediction == c
                    t = target == c

                    tp = int(
                        (p & t).sum().item()
                    )

                    fp = int(
                        (p & ~t).sum().item()
                    )

                    fn = int(
                        (~p & t).sum().item()
                    )

                    aggregate[c]["target"] += int(
                        t.sum().item()
                    )

                    aggregate[c]["prediction"] += int(
                        p.sum().item()
                    )

                    aggregate[c]["tp"] += tp
                    aggregate[c]["fp"] += fp
                    aggregate[c]["fn"] += fn

                    record[
                        f"class_{c}_dice"
                    ] = class_dice[c]

                case_records.append(
                    record
                )

                if (
                    case_index <= 5
                    or case_index % 25 == 0
                    or case_index == len(cohort)
                ):
                    print(
                        f"  [{case_index:03d}/{len(cohort)}] "
                        f"{study} | {series} | "
                        f"pred_fg={prediction_fg} | "
                        f"target_fg={target_fg} | "
                        f"Dice={mean_dice:.6f} | "
                        f"labels=[{record['unique_prediction_labels']}]"
                    )

            except Exception as exc:
                failed.append({
                    "case_index": case_index,
                    "study_id": study,
                    "series_id": series,
                    "status": "ERROR",
                    "error": repr(exc),
                })

                case_records.append(
                    failed[-1]
                )

                print(
                    f"  [{case_index:03d}/{len(cohort)}] "
                    f"{study} | {series} | ERROR: {exc}"
                )

    # -------------------------------------------------------------------------
    # AGGREGATE
    # -------------------------------------------------------------------------
    section("PART 37 OVERALL DIAGNOSIS")

    successful = len(case_records) - len(failed)

    total_target_fg = sum(
        aggregate[c]["target"]
        for c in FOREGROUND
    )

    total_prediction_fg = sum(
        aggregate[c]["prediction"]
        for c in FOREGROUND
    )

    aggregate_class = []

    class_dice_values = []

    for c in FOREGROUND:

        a = aggregate[c]

        denominator = (
            2 * a["tp"]
            + a["fp"]
            + a["fn"]
        )

        d = (
            float("nan")
            if denominator == 0
            else 2.0 * a["tp"] / denominator
        )

        if not math.isnan(d):
            class_dice_values.append(d)

        p = (
            a["tp"]
            / (a["tp"] + a["fp"])
            if a["tp"] + a["fp"]
            else 0.0
        )

        r = (
            a["tp"]
            / (a["tp"] + a["fn"])
            if a["tp"] + a["fn"]
            else 0.0
        )

        aggregate_class.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "target_voxels": a["target"],
            "prediction_voxels": a["prediction"],
            "tp": a["tp"],
            "fp": a["fp"],
            "fn": a["fn"],
            "dice": d,
            "precision": p,
            "recall": r,
        })

    aggregate_dice = (
        float(np.mean(class_dice_values))
        if class_dice_values
        else float("nan")
    )

    fg_ratio = (
        total_prediction_fg
        / total_target_fg
        if total_target_fg
        else float("nan")
    )

    source_api_incompatible = any(
        item["positional_args"] != 2
        for item in metric_calls
    )

    checkpoint_dice = (
        checkpoint.get("best_val_dice")
        if isinstance(checkpoint, dict)
        else None
    )

    if (
        source_api_incompatible
        and total_prediction_fg == 0
    ):
        diagnosis = (
            "HISTORICAL_PART15_METRIC_API_MISMATCH_CONFIRMED_"
            "AND_SAVED_CHECKPOINT_FOREGROUND_COLLAPSE"
        )
    elif source_api_incompatible:
        diagnosis = (
            "HISTORICAL_PART15_METRIC_API_MISMATCH_REQUIRES_"
            "HISTORICAL_RUNTIME_RECONSTRUCTION"
        )
    elif total_prediction_fg == 0:
        diagnosis = (
            "SAVED_CHECKPOINT_FOREGROUND_COLLAPSE"
        )
    else:
        diagnosis = (
            "NO_COMPLETE_CHECKPOINT_COLLAPSE"
        )

    print(
        f"Successful cases                     : "
        f"{successful}"
    )

    print(
        f"Failed cases                         : "
        f"{len(failed)}"
    )

    print(
        f"Target foreground voxels             : "
        f"{total_target_fg:,}"
    )

    print(
        f"Predicted foreground voxels          : "
        f"{total_prediction_fg:,}"
    )

    print(
        f"Prediction / target foreground ratio : "
        f"{fg_ratio:.6f}"
    )

    print(
        f"Reconstructed foreground Dice        : "
        f"{aggregate_dice:.6f}"
    )

    print(
        f"Historical Part 15 Dice              : "
        f"{recorded_best if 'recorded_best' in locals() else 'N/A'}"
    )

    print(
        f"Checkpoint stored best Dice          : "
        f"{checkpoint_dice}"
    )

    print(
        f"Historical metric API incompatible   : "
        f"{source_api_incompatible}"
    )

    print(
        f"Saved checkpoint foreground collapse : "
        f"{total_prediction_fg == 0}"
    )

    print(
        f"DIAGNOSIS                            : "
        f"{diagnosis}"
    )

    # -------------------------------------------------------------------------
    # CLASS SUMMARY
    # -------------------------------------------------------------------------
    section("CLASS-WISE CHECKPOINT FORENSICS")

    class_df = pd.DataFrame(
        aggregate_class
    )

    print(
        class_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # SOURCE SNIPPET
    # -------------------------------------------------------------------------
    section("RELEVANT PART 15 SOURCE SNIPPET")

    lines = part15_text.splitlines()

    relevant_numbers = sorted(
        set(
            [item["line"] for item in metric_calls]
            + [item["line"] for item in save_calls]
        )
    )

    shown = set()

    for line_no in relevant_numbers:
        start = max(
            1,
            line_no - 4,
        )

        end = min(
            len(lines),
            line_no + 6,
        )

        if (start, end) in shown:
            continue

        shown.add(
            (start, end)
        )

        print(
            f"\n--- Part 15 lines {start}-{end} ---"
        )

        for n in range(
            start,
            end + 1,
        ):
            print(
                f"{n:04d}: {lines[n - 1]}"
            )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    section("SAVING PART 37 OUTPUTS")

    case_df = pd.DataFrame(
        case_records
    )

    case_df.to_csv(
        CASE_CSV,
        index=False,
    )

    class_df.to_csv(
        CLASS_CSV,
        index=False,
    )

    source_df.to_csv(
        SOURCE_CSV,
        index=False,
    )

    summary = {
        "phase": "Phase 4 - Part 37",
        "description": (
            "RSNA-only Part 15 metric-path and checkpoint-origin "
            "forensic audit"
        ),
        "validation_cases_requested": VAL_CASES,
        "successful_cases": successful,
        "failed_cases": len(failed),
        "part15_source_metric_calls": metric_calls,
        "part15_source_checkpoint_save_calls": save_calls,
        "historical_metric_api_incompatible": bool(
            source_api_incompatible
        ),
        "part15_recorded_best_dice": (
            float(recorded_best)
            if "recorded_best" in locals()
            and not math.isnan(recorded_best)
            else None
        ),
        "checkpoint_stored_best_dice": (
            float(checkpoint_dice)
            if isinstance(
                checkpoint_dice,
                (int, float),
            )
            else None
        ),
        "reconstructed_foreground_dice": (
            float(aggregate_dice)
            if not math.isnan(aggregate_dice)
            else None
        ),
        "target_foreground_voxels": int(
            total_target_fg
        ),
        "prediction_foreground_voxels": int(
            total_prediction_fg
        ),
        "prediction_target_foreground_ratio": float(
            fg_ratio
        ),
        "saved_checkpoint_foreground_collapse": (
            total_prediction_fg == 0
        ),
        "diagnosis": diagnosis,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_metadata": checkpoint_info,
        "class_summary": aggregate_class,
        "failed_cases": failed,
        "training_performed": False,
        "model_weights_modified": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "spider_used": False,
        "rsna_test_set_used": False,
        "pseudo_masks_are_manual_ground_truth": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = [
        "PHASE 4 - PART 37",
        "PART 15 METRIC-PATH / CHECKPOINT-ORIGIN FORENSIC AUDIT",
        "=" * 78,
        "",
        f"Successful cases: {successful}",
        f"Failed cases: {len(failed)}",
        f"Part 15 recorded best Dice: "
        f"{summary['part15_recorded_best_dice']}",
        f"Checkpoint stored best Dice: "
        f"{summary['checkpoint_stored_best_dice']}",
        f"Reconstructed foreground Dice: "
        f"{summary['reconstructed_foreground_dice']}",
        f"Target foreground voxels: {total_target_fg}",
        f"Predicted foreground voxels: {total_prediction_fg}",
        f"Prediction/target foreground ratio: {fg_ratio}",
        f"Historical metric API incompatible: "
        f"{source_api_incompatible}",
        f"Saved checkpoint foreground collapse: "
        f"{total_prediction_fg == 0}",
        f"DIAGNOSIS: {diagnosis}",
        "",
        "CLASS SUMMARY",
        "-" * 78,
        class_df.to_string(index=False),
        "",
        "SOURCE METRIC CALLS",
        "-" * 78,
        json.dumps(
            metric_calls,
            indent=2,
        ),
        "",
        "CHECKPOINT SAVE CALLS",
        "-" * 78,
        json.dumps(
            save_calls,
            indent=2,
        ),
    ]

    REPORT_TXT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"Checkpoint inventory CSV              : "
        f"{CHECKPOINT_CSV}"
    )

    print(
        f"Source findings CSV                   : "
        f"{SOURCE_CSV}"
    )

    print(
        f"Case reconstruction CSV               : "
        f"{CASE_CSV}"
    )

    print(
        f"Class metric summary CSV              : "
        f"{CLASS_CSV}"
    )

    print(
        f"Summary JSON                          : "
        f"{SUMMARY_JSON}"
    )

    print(
        f"Report TXT                            : "
        f"{REPORT_TXT}"
    )

    section("PHASE 4 - PART 37 COMPLETE")

    print(
        "✓ Part 15 source metric path inspected."
    )

    print(
        "✓ Part 15 checkpoint-save path inspected."
    )

    print(
        "✓ Available checkpoint files inventoried."
    )

    print(
        "✓ Checkpoint metadata inspected."
    )

    print(
        "✓ Exact 100-case validation cohort reconstructed."
    )

    print(
        "✓ Saved checkpoint independently evaluated."
    )

    print(
        "✓ Class-wise checkpoint metrics reconstructed."
    )

    print(
        "✓ Historical/current API compatibility recorded."
    )

    print(
        "✓ No training performed."
    )

    print(
        "✓ No model weights modified."
    )

    print(
        "✓ No optimizer created."
    )

    print(
        "✓ No optimizer step performed."
    )

    print(
        "✓ SPIDER not used."
    )

    print(
        "✓ RSNA test set not used."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 78)
        print("PART 37 ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
