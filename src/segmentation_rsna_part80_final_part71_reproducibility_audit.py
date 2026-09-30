"""
PART 80 — FINAL PART 71 REPRODUCIBILITY AUDIT
==============================================

Purpose
-------
Part 79 reproduced the low ~0.0014 FG Dice result, while the historical
Part 71 class-balanced run reported ~0.0440 FG Dice. Therefore we must
audit the implementation before doing any more training.

IMPORTANT:
- This script is an AUDIT, not a training experiment.
- It performs NO optimizer step.
- It performs NO backward pass.
- It does NOT modify any checkpoint.
- It does NOT create a new model checkpoint.
- It uses the exact Part 15 initialization only for evaluation.
- It inspects the actual Part 71 source code when available.
- It compares the current Part 79 loss implementation against the
  historical Part 71 implementation.
- It checks the actual first 100/50 cohort preprocessing and mask statistics.
- It compares historical Part 71 report files if they exist.

Run:
python ".\\src\\segmentation_rsna_part80_final_part71_reproducibility_audit.py"

After this audit, do not automatically launch another training run.
Use the report to identify the exact discrepancy first.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)
SRC_DIR = PROJECT_ROOT / "src"

PART9_PATH = SRC_DIR / "segmentation_rsna_part9_3d_dataset_loader.py"
PART11_PATH = SRC_DIR / "segmentation_rsna_part11_controlled_pilot_training.py"

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"
INIT_CKPT = (
    PART15_DIR
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

# Historical Part 71 output.
PART71_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part71_class_balanced_dicece_training"
)

# Current Part 79 output.
PART79_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part79_part71_final_reproducibility_check"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part80_final_part71_reproducibility_audit"
)

REPORT_DIR = OUT_DIR / "reports"


# =====================================================================
# LOCKED VALUES
# =====================================================================

SEED = 42
TRAIN_N = 100
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1
NUM_CLASSES = 6

PART15_EXPECTED_SHA256 = (
    "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
)

CLASS_WEIGHTS = torch.tensor(
    [
        0.05,
        0.722595,
        0.935995,
        1.017003,
        1.178304,
        1.146102,
    ],
    dtype=torch.float32,
)

PART71_REFERENCE_E3 = 0.044026


# =====================================================================
# UTILITIES
# =====================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""


def find_source_candidates() -> List[Path]:
    candidates = []

    # Common naming variants from the established project.
    patterns = [
        "*part71*.py",
        "*Part71*.py",
        "*class_balanced*dicece*.py",
        "*class*balanced*.py",
    ]

    seen = set()

    for pattern in patterns:
        for p in SRC_DIR.glob(pattern):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                candidates.append(p)

    return sorted(candidates)


def find_files_recursive(
    root: Path,
    patterns: List[str],
) -> List[Path]:

    if not root.exists():
        return []

    found = []
    seen = set()

    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file() and p.resolve() not in seen:
                seen.add(p.resolve())
                found.append(p)

    return sorted(found)


def extract_text_context(
    text: str,
    terms: List[str],
    radius: int = 500,
) -> Dict[str, List[str]]:

    output = {}

    for term in terms:

        matches = []

        for match in re.finditer(
            re.escape(term),
            text,
            flags=re.IGNORECASE,
        ):

            start = max(
                0,
                match.start() - radius,
            )

            end = min(
                len(text),
                match.end() + radius,
            )

            matches.append(
                text[start:end]
            )

        output[term] = matches[:5]

    return output


def tensor_stats(mask: torch.Tensor) -> Dict[str, Any]:

    mask = mask.long()

    counts = {}

    for c in range(NUM_CLASSES):
        count = int(
            (mask == c).sum().item()
        )
        counts[str(c)] = count

    fg = int(
        (mask > 0).sum().item()
    )

    total = int(
        mask.numel()
    )

    return {
        "shape": list(mask.shape),
        "total_voxels": total,
        "foreground_voxels": fg,
        "foreground_fraction": (
            fg / total
            if total
            else 0.0
        ),
        "class_counts": counts,
    }


# =====================================================================
# PATH AUDIT
# =====================================================================

def audit_paths() -> Dict[str, Any]:

    paths = {
        "project_root": PROJECT_ROOT,
        "part9": PART9_PATH,
        "part11": PART11_PATH,
        "train_cohort": TRAIN_COHORT,
        "val_cohort": VAL_COHORT,
        "part15_init": INIT_CKPT,
        "part71_output": PART71_DIR,
        "part79_output": PART79_DIR,
    }

    result = {}

    print("\n[PATH AUDIT]")

    for name, path in paths.items():

        exists = path.exists()

        result[name] = {
            "path": str(path),
            "exists": exists,
        }

        print(
            f"  {name}: "
            f"{'FOUND' if exists else 'MISSING'} -> {path}"
        )

    return result


# =====================================================================
# PART 15 CHECKPOINT AUDIT
# =====================================================================

def audit_part15_checkpoint() -> Dict[str, Any]:

    print("\n[PART 15 CHECKPOINT AUDIT]")

    actual_hash = sha256_file(
        INIT_CKPT
    )

    hash_match = (
        actual_hash
        == PART15_EXPECTED_SHA256
    )

    print(
        f"  SHA256: {actual_hash}"
    )

    print(
        f"  Expected match: {hash_match}"
    )

    checkpoint = torch.load(
        INIT_CKPT,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())

        print(
            f"  Checkpoint type: full dictionary"
        )

        print(
            f"  Keys: {keys}"
        )

        has_model_state = (
            "model_state_dict"
            in checkpoint
        )

    else:
        keys = []
        has_model_state = False

    print(
        f"  Contains model_state_dict: "
        f"{has_model_state}"
    )

    model_state = (
        checkpoint["model_state_dict"]
        if has_model_state
        else checkpoint
    )

    model_keys = list(
        model_state.keys()
    )

    print(
        f"  Model parameter tensors: "
        f"{len(model_keys)}"
    )

    return {
        "sha256": actual_hash,
        "sha256_matches_expected": hash_match,
        "checkpoint_keys": keys,
        "has_model_state_dict": has_model_state,
        "model_parameter_tensor_count": len(model_keys),
    }


# =====================================================================
# SOURCE AUDIT
# =====================================================================

def audit_part71_source() -> Dict[str, Any]:

    print("\n[PART 71 SOURCE AUDIT]")

    candidates = find_source_candidates()

    result = {
        "candidate_files": [],
        "selected_files": [],
        "loss_context": {},
        "preprocess_context": {},
        "metric_context": {},
        "training_context": {},
    }

    if not candidates:
        print(
            "  No Part 71 source file found by filename search."
        )
        return result

    for path in candidates:

        print(
            f"  Candidate: {path}"
        )

        text = safe_read_text(path)

        result["candidate_files"].append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

        result["selected_files"].append(
            str(path)
        )

        result["loss_context"][str(path)] = (
            extract_text_context(
                text,
                [
                    "DiceCELoss",
                    "class_weights",
                    "weight",
                    "CrossEntropyLoss",
                    "cross_entropy",
                    "lambda_dice",
                    "lambda_ce",
                ],
            )
        )

        result["preprocess_context"][str(path)] = (
            extract_text_context(
                text,
                [
                    "centered",
                    "center_crop",
                    "CROP_SHAPE",
                    "crop_shape",
                    "R2",
                    "pseudo",
                    "preprocess_case",
                    "load_tensor_case",
                ],
            )
        )

        result["metric_context"][str(path)] = (
            extract_text_context(
                text,
                [
                    "dice_from_prediction",
                    "FGDice",
                    "foreground",
                    "argmax",
                    "classwise",
                ],
            )
        )

        result["training_context"][str(path)] = (
            extract_text_context(
                text,
                [
                    "EPOCHS",
                    "TRAIN_N",
                    "VAL_N",
                    "learning_rate",
                    "weight_decay",
                    "AdamW",
                    "optimizer",
                    "seed",
                    "random",
                ],
            )
        )

    return result


# =====================================================================
# HISTORICAL PART 71 REPORT AUDIT
# =====================================================================

def audit_part71_outputs() -> Dict[str, Any]:

    print("\n[PART 71 OUTPUT AUDIT]")

    files = find_files_recursive(
        PART71_DIR,
        [
            "*.csv",
            "*.json",
            "*.txt",
        ],
    )

    result = {
        "files": [],
        "relevant_files": [],
        "reference_values": [],
    }

    if not files:
        print(
            "  No historical Part 71 output files found."
        )
        return result

    keywords = [
        "learning",
        "trajectory",
        "summary",
        "classwise",
        "report",
        "comparison",
        "dice",
    ]

    for path in files:

        relative = str(
            path.relative_to(PART71_DIR)
        )

        result["files"].append(
            relative
        )

        lower = relative.lower()

        if any(
            key in lower
            for key in keywords
        ):
            result["relevant_files"].append(
                relative
            )

        text = safe_read_text(path)

        # Search for historical E3/FG Dice values.
        patterns = [
            r"FGDice[^0-9]*([0-9]+\.[0-9]+)",
            r"fg_dice[^0-9]*([0-9]+\.[0-9]+)",
            r"foreground[^0-9]*dice[^0-9]*([0-9]+\.[0-9]+)",
        ]

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                try:
                    value = float(
                        match.group(1)
                    )
                except Exception:
                    continue

                if 0.0 <= value <= 1.0:

                    result["reference_values"].append(
                        {
                            "file": relative,
                            "value": value,
                            "context": text[
                                max(
                                    0,
                                    match.start() - 120,
                                ):
                                min(
                                    len(text),
                                    match.end() + 120,
                                )
                            ],
                        }
                    )

    # Print likely values close to the known Part 71 reference.
    candidates = [
        x
        for x in result["reference_values"]
        if abs(
            x["value"]
            - PART71_REFERENCE_E3
        ) < 0.01
    ]

    print(
        f"  Files found: {len(files)}"
    )

    print(
        f"  Values close to Part 71 reference "
        f"({PART71_REFERENCE_E3}): {len(candidates)}"
    )

    for item in candidates[:10]:

        print(
            f"    {item['file']}: "
            f"{item['value']}"
        )

    return result


# =====================================================================
# ACTUAL CURRENT PIPELINE AUDIT
# =====================================================================

def as_dhw(x: torch.Tensor) -> torch.Tensor:

    x = x.detach().cpu()

    while x.ndim > 3:

        singleton = None

        for i, size in enumerate(x.shape):

            if size == 1:
                singleton = i
                break

        if singleton is None:
            raise ValueError(
                f"Cannot reduce {tuple(x.shape)} to DHW"
            )

        x = x.squeeze(singleton)

    if x.ndim != 3:
        raise ValueError(
            f"Expected DHW; got {tuple(x.shape)}"
        )

    return x


def center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
):

    d, h, w = image.shape
    cd, ch, cw = CROP_SHAPE

    z0 = (d - cd) // 2
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    return (
        image[
            z0:z0 + cd,
            y0:y0 + ch,
            x0:x0 + cw,
        ],
        mask[
            z0:z0 + cd,
            y0:y0 + ch,
            x0:x0 + cw,
        ],
    )


def load_audit_cases(
    df: pd.DataFrame,
    part9,
    part11,
    split_name: str,
) -> List[Dict[str, Any]]:

    cases = []

    print(
        f"\n[PIPELINE AUDIT — {split_name.upper()}]"
    )

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        image = as_dhw(
            loaded[0]
        ).float()

        mask = as_dhw(
            loaded[1]
        ).long()

        image_crop, mask_crop = center_crop(
            image,
            mask,
        )

        case = {
            "index": i,
            "case_id": str(
                row.get(
                    "study_id",
                    row.get(
                        "series_id",
                        row.get(
                            "id",
                            i,
                        ),
                    ),
                )
            ),
            "image_shape_before_crop": list(
                image.shape
            ),
            "mask_shape_before_crop": list(
                mask.shape
            ),
            "image_shape_after_crop": list(
                image_crop.shape
            ),
            "mask_shape_after_crop": list(
                mask_crop.shape
            ),
            "mask_stats": tensor_stats(
                mask_crop
            ),
        }

        cases.append(
            case
        )

    fg_values = [
        x["mask_stats"]["foreground_voxels"]
        for x in cases
    ]

    class_totals = {
        str(c): 0
        for c in range(NUM_CLASSES)
    }

    for case in cases:

        for c, value in case[
            "mask_stats"
        ]["class_counts"].items():

            class_totals[c] += value

    print(
        f"  Cases loaded: {len(cases)}"
    )

    print(
        f"  Mean crop foreground voxels: "
        f"{np.mean(fg_values):.4f}"
    )

    print(
        f"  Median crop foreground voxels: "
        f"{np.median(fg_values):.4f}"
    )

    print(
        f"  Class totals: {class_totals}"
    )

    return cases


def compare_pipeline_statistics(
    train_cases,
    val_cases,
) -> Dict[str, Any]:

    result = {}

    for split_name, cases in [
        ("train", train_cases),
        ("validation", val_cases),
    ]:

        foreground = np.array(
            [
                c["mask_stats"]["foreground_voxels"]
                for c in cases
            ],
            dtype=np.float64,
        )

        fractions = np.array(
            [
                c["mask_stats"]["foreground_fraction"]
                for c in cases
            ],
            dtype=np.float64,
        )

        class_means = {}

        for c in range(NUM_CLASSES):

            values = [
                case[
                    "mask_stats"
                ]["class_counts"][str(c)]
                for case in cases
            ]

            class_means[str(c)] = float(
                np.mean(values)
            )

        result[split_name] = {
            "n": len(cases),
            "mean_foreground_voxels": float(
                foreground.mean()
            ),
            "median_foreground_voxels": float(
                np.median(foreground)
            ),
            "mean_foreground_fraction": float(
                fractions.mean()
            ),
            "mean_class_voxels": class_means,
        }

    return result


# =====================================================================
# LOSS FORMULATION AUDIT
# =====================================================================

class CurrentCompatibilityDiceCE(torch.nn.Module):
    """
    Exactly the compatibility formulation used in Parts 78/79:

        MONAI Dice
        +
        weighted PyTorch cross entropy

    This is intentionally NOT used for training in Part 80.
    """

    def __init__(self, device):

        super().__init__()

        self.register_buffer(
            "weights",
            CLASS_WEIGHTS.to(device),
        )

        self.dice = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
            squared_pred=False,
            reduction="mean",
            lambda_dice=1.0,
            lambda_ce=0.0,
        )

    def forward(
        self,
        logits,
        target,
    ):

        dice_value = self.dice(
            logits,
            target,
        )

        ce_target = target[:, 0].long()

        ce_value = F.cross_entropy(
            logits,
            ce_target,
            weight=self.weights,
            reduction="mean",
        )

        return dice_value + ce_value


def audit_loss_formulations(
    device,
) -> Dict[str, Any]:

    print("\n[LOSS FORMULATION AUDIT]")

    torch.manual_seed(79)

    # Small synthetic tensor to compare formulation behavior.
    logits = torch.randn(
        1,
        NUM_CLASSES,
        4,
        8,
        8,
        device=device,
    )

    target = torch.randint(
        0,
        NUM_CLASSES,
        (
            1,
            1,
            4,
            8,
            8,
        ),
        device=device,
    )

    current_loss = CurrentCompatibilityDiceCE(
        device
    )

    current_value = current_loss(
        logits,
        target,
    )

    # Native unweighted DiceCE is included only as a diagnostic reference.
    native = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        sigmoid=False,
        squared_pred=False,
        reduction="mean",
    )

    native_value = native(
        logits,
        target,
    )

    weighted_ce = F.cross_entropy(
        logits,
        target[:, 0].long(),
        weight=CLASS_WEIGHTS.to(device),
        reduction="mean",
    )

    unweighted_ce = F.cross_entropy(
        logits,
        target[:, 0].long(),
        reduction="mean",
    )

    result = {
        "current_compatibility_dicece": float(
            current_value.item()
        ),
        "native_monai_unweighted_dicece": float(
            native_value.item()
        ),
        "weighted_ce": float(
            weighted_ce.item()
        ),
        "unweighted_ce": float(
            unweighted_ce.item()
        ),
        "class_weights": CLASS_WEIGHTS.tolist(),
    }

    print(
        f"  Current compatibility DiceCE: "
        f"{result['current_compatibility_dicece']:.6f}"
    )

    print(
        f"  Native unweighted MONAI DiceCE: "
        f"{result['native_monai_unweighted_dicece']:.6f}"
    )

    print(
        f"  Weighted CE: "
        f"{result['weighted_ce']:.6f}"
    )

    print(
        f"  Unweighted CE: "
        f"{result['unweighted_ce']:.6f}"
    )

    return result


# =====================================================================
# PART 79 OUTPUT AUDIT
# =====================================================================

def audit_part79_outputs() -> Dict[str, Any]:

    print("\n[PART 79 OUTPUT AUDIT]")

    files = find_files_recursive(
        PART79_DIR,
        [
            "*.csv",
            "*.json",
            "*.txt",
        ],
    )

    result = {
        "files": [],
        "trajectory": [],
        "final_metrics": [],
    }

    for path in files:

        result["files"].append(
            str(
                path.relative_to(
                    PART79_DIR
                )
            )
        )

        if path.name == (
            "part79_learning_trajectory.csv"
        ):

            try:
                df = pd.read_csv(
                    path
                )

                result["trajectory"] = (
                    df.to_dict(
                        orient="records"
                    )
                )

            except Exception as exc:

                result["trajectory_error"] = (
                    str(exc)
                )

        if path.name == (
            "part79_final_metrics.csv"
        ):

            try:
                df = pd.read_csv(
                    path
                )

                result["final_metrics"] = (
                    df.to_dict(
                        orient="records"
                    )
                )

            except Exception as exc:

                result["final_metrics_error"] = (
                    str(exc)
                )

    print(
        f"  Output files found: "
        f"{len(result['files'])}"
    )

    if result["trajectory"]:

        print(
            "  Part 79 trajectory:"
        )

        for row in result["trajectory"]:

            print(
                f"    E{row['epoch']}: "
                f"FGDice={row['fg_dice']:.6f}"
            )

    return result


# =====================================================================
# DECISION LOGIC
# =====================================================================

def derive_audit_decision(
    path_audit,
    checkpoint_audit,
    source_audit,
    part71_outputs,
    pipeline_stats,
    loss_audit,
    part79_outputs,
) -> Dict[str, Any]:

    findings = []

    if not checkpoint_audit[
        "sha256_matches_expected"
    ]:
        findings.append(
            "PART15_CHECKPOINT_HASH_MISMATCH"
        )

    if not checkpoint_audit[
        "has_model_state_dict"
    ]:
        findings.append(
            "PART15_CHECKPOINT_FORMAT_UNEXPECTED"
        )

    if not source_audit[
        "candidate_files"
    ]:
        findings.append(
            "PART71_SOURCE_FILE_NOT_FOUND_BY_FILENAME"
        )

    if not part71_outputs[
        "files"
    ]:
        findings.append(
            "PART71_OUTPUT_FILES_NOT_FOUND"
        )

    if pipeline_stats:

        for split_name, stats in pipeline_stats.items():

            if tuple(
                [32, 64, 64]
            ) != tuple(
                CROP_SHAPE
            ):
                findings.append(
                    f"{split_name.upper()}_CROP_SHAPE_MISMATCH"
                )

    historical_values = [
        x["value"]
        for x in part71_outputs[
            "reference_values"
        ]
        if abs(
            x["value"]
            - PART71_REFERENCE_E3
        ) < 0.01
    ]

    current_values = []

    for row in part79_outputs.get(
        "trajectory",
        [],
    ):

        if int(
            row["epoch"]
        ) == 3:

            current_values.append(
                float(
                    row["fg_dice"]
                )
            )

    current_e3 = (
        current_values[-1]
        if current_values
        else None
    )

    discrepancy = (
        abs(
            current_e3
            - PART71_REFERENCE_E3
        )
        if current_e3 is not None
        else None
    )

    # We intentionally do not declare the root cause from this audit
    # unless the historical source/output evidence supports it.
    if discrepancy is not None and discrepancy > 0.01:

        findings.append(
            "PART71_VS_PART79_METRIC_DISCREPANCY_CONFIRMED"
        )

    if (
        historical_values
        and current_e3 is not None
        and discrepancy is not None
        and discrepancy > 0.01
    ):

        status = (
            "AUDIT_FOUND_HISTORICAL_PART71_REFERENCE_BUT_CURRENT_PIPELINE_DIFFERS"
        )

    elif not source_audit[
        "candidate_files"
    ]:

        status = (
            "AUDIT_INCOMPLETE_PART71_SOURCE_NOT_LOCATED"
        )

    else:

        status = (
            "AUDIT_COMPLETE_REVIEW_SOURCE_CONTEXT_BEFORE_ANY_TRAINING"
        )

    return {
        "status": status,
        "findings": findings,
        "historical_reference_values_near_0.044026": historical_values[:20],
        "part79_e3_fg_dice": current_e3,
        "part71_reference_e3_fg_dice": PART71_REFERENCE_E3,
        "absolute_e3_difference": discrepancy,
        "recommendation": (
            "Do not launch another ablation automatically. "
            "Use the source/report evidence to identify the implementation "
            "difference before any further training."
        ),
    }


# =====================================================================
# REPORT WRITING
# =====================================================================

def write_reports(report: Dict[str, Any]) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        REPORT_DIR
        / "part80_audit_summary.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Flatten the key decision fields.
    csv_row = {
        "status": report[
            "decision"
        ]["status"],
        "part71_reference_e3": report[
            "decision"
        ]["part71_reference_e3_fg_dice"],
        "part79_e3": report[
            "decision"
        ]["part79_e3_fg_dice"],
        "absolute_difference": report[
            "decision"
        ]["absolute_e3_difference"],
        "part15_hash_matches": report[
            "checkpoint_audit"
        ]["sha256_matches_expected"],
        "part15_has_model_state_dict": report[
            "checkpoint_audit"
        ]["has_model_state_dict"],
        "part71_source_candidates": len(
            report[
                "source_audit"
            ]["candidate_files"]
        ),
        "part71_output_files": len(
            report[
                "part71_outputs"
            ]["files"]
        ),
    }

    csv_path = (
        REPORT_DIR
        / "part80_audit_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                csv_row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            csv_row
        )

    txt_lines = []

    txt_lines.append(
        "PART 80 — FINAL PART 71 REPRODUCIBILITY AUDIT"
    )
    txt_lines.append(
        "=" * 70
    )
    txt_lines.append("")
    txt_lines.append(
        f"STATUS: {report['decision']['status']}"
    )
    txt_lines.append("")
    txt_lines.append(
        "IMPORTANT: This was an audit only. "
        "No optimizer step, backward pass, or checkpoint modification "
        "was performed."
    )
    txt_lines.append("")
    txt_lines.append(
        "PART 15 CHECKPOINT"
    )
    txt_lines.append(
        f"SHA256: {report['checkpoint_audit']['sha256']}"
    )
    txt_lines.append(
        f"SHA256 match: "
        f"{report['checkpoint_audit']['sha256_matches_expected']}"
    )
    txt_lines.append(
        f"Has model_state_dict: "
        f"{report['checkpoint_audit']['has_model_state_dict']}"
    )
    txt_lines.append("")
    txt_lines.append(
        "PART 71 REFERENCE"
    )
    txt_lines.append(
        f"E3 FG Dice reference: "
        f"{PART71_REFERENCE_E3:.6f}"
    )
    txt_lines.append(
        f"Historical values near reference: "
        f"{report['decision']['historical_reference_values_near_0.044026']}"
    )
    txt_lines.append("")
    txt_lines.append(
        "PART 79 COMPARISON"
    )
    txt_lines.append(
        f"E3 FG Dice: "
        f"{report['decision']['part79_e3_fg_dice']}"
    )
    txt_lines.append(
        f"Absolute difference: "
        f"{report['decision']['absolute_e3_difference']}"
    )
    txt_lines.append("")
    txt_lines.append(
        "PIPELINE STATISTICS"
    )

    for split, stats in report[
        "pipeline_statistics"
    ].items():

        txt_lines.append(
            f"{split}: "
            f"n={stats['n']}, "
            f"mean_fg_voxels="
            f"{stats['mean_foreground_voxels']:.4f}, "
            f"median_fg_voxels="
            f"{stats['median_foreground_voxels']:.4f}, "
            f"mean_fg_fraction="
            f"{stats['mean_foreground_fraction']:.8f}"
        )

    txt_lines.append("")
    txt_lines.append(
        "FINDINGS"
    )

    for finding in report[
        "decision"
    ]["findings"]:

        txt_lines.append(
            f"- {finding}"
        )

    txt_lines.append("")
    txt_lines.append(
        "RECOMMENDATION"
    )
    txt_lines.append(
        report[
            "decision"
        ]["recommendation"]
    )

    (
        REPORT_DIR
        / "part80_report.txt"
    ).write_text(
        "\n".join(
            txt_lines
        ),
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 78)
    print(
        "PART 80 — FINAL PART 71 REPRODUCIBILITY AUDIT"
    )
    print("=" * 78)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_audit = audit_paths()

    if not INIT_CKPT.exists():
        raise FileNotFoundError(
            f"Missing Part 15 checkpoint: {INIT_CKPT}"
        )

    checkpoint_audit = (
        audit_part15_checkpoint()
    )

    source_audit = (
        audit_part71_source()
    )

    part71_outputs = (
        audit_part71_outputs()
    )

    part79_outputs = (
        audit_part79_outputs()
    )

    # Load pipeline only after the audit of source/output files.
    part9 = load_module_from_file(
        "part9_part80",
        PART9_PATH,
    )

    part11 = load_module_from_file(
        "part11_part80",
        PART11_PATH,
    )

    train_df = pd.read_csv(
        TRAIN_COHORT
    ).iloc[:TRAIN_N].reset_index(
        drop=True
    )

    val_df = pd.read_csv(
        VAL_COHORT
    ).iloc[:VAL_N].reset_index(
        drop=True
    )

    train_cases = load_audit_cases(
        train_df,
        part9,
        part11,
        "train",
    )

    val_cases = load_audit_cases(
        val_df,
        part9,
        part11,
        "validation",
    )

    pipeline_statistics = (
        compare_pipeline_statistics(
            train_cases,
            val_cases,
        )
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    loss_audit = (
        audit_loss_formulations(
            device
        )
    )

    decision = derive_audit_decision(
        path_audit=path_audit,
        checkpoint_audit=checkpoint_audit,
        source_audit=source_audit,
        part71_outputs=part71_outputs,
        pipeline_stats=pipeline_statistics,
        loss_audit=loss_audit,
        part79_outputs=part79_outputs,
    )

    report = {
        "part": 80,
        "title": (
            "Final Part 71 Reproducibility Audit"
        ),
        "audit_only": True,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "checkpoint_modifications": 0,
        "device": str(device),
        "locked_parameters": {
            "seed": SEED,
            "train_n": TRAIN_N,
            "val_n": VAL_N,
            "full_shape": FULL_SHAPE,
            "crop_shape": CROP_SHAPE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": NUM_CLASSES,
            "class_weights": CLASS_WEIGHTS.tolist(),
        },
        "path_audit": path_audit,
        "checkpoint_audit": checkpoint_audit,
        "source_audit": source_audit,
        "part71_outputs": part71_outputs,
        "part79_outputs": part79_outputs,
        "pipeline_statistics": pipeline_statistics,
        "loss_audit": loss_audit,
        "decision": decision,
    }

    write_reports(
        report
    )

    print("")
    print("=" * 78)
    print(
        "PART 80 AUDIT COMPLETE"
    )
    print("=" * 78)

    print(
        f"Status: {decision['status']}"
    )

    print(
        f"Part 71 reference E3: "
        f"{PART71_REFERENCE_E3:.6f}"
    )

    print(
        f"Part 79 E3: "
        f"{decision['part79_e3_fg_dice']}"
    )

    print(
        f"Absolute difference: "
        f"{decision['absolute_e3_difference']}"
    )

    print(
        "\nFindings:"
    )

    for finding in decision[
        "findings"
    ]:

        print(
            f"  - {finding}"
        )

    print(
        "\nReports:"
    )

    for path in sorted(
        REPORT_DIR.glob("part80_*")
    ):

        print(
            f"  {path}"
        )

    print(
        "\nNO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
