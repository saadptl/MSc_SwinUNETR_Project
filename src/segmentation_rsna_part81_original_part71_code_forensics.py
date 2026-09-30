"""
PART 81 — ORIGINAL PART 71 CODE FORENSICS
==========================================

Purpose
-------
Part 80 proved that the historical Part 71 E3 foreground Dice (~0.044026)
does not match the Part 79 compatibility reproduction (~0.001406).

Part 81 performs CODE FORENSICS ONLY.

It does NOT:
- train,
- perform backward(),
- perform optimizer.step(),
- modify checkpoints,
- launch a GPU training experiment.

It inspects the ORIGINAL Part 71 source and historical outputs, then:
1. Extracts the exact Part 71 loss implementation.
2. Extracts DiceCELoss arguments and class-weight handling.
3. Extracts preprocessing/cropping.
4. Extracts validation metric implementation.
5. Extracts checkpoint saving/loading.
6. Extracts cohort/seed/hyperparameter configuration.
7. Locates the Part 71 epoch-3 checkpoint.
8. If possible, loads the saved Part 71 epoch-3 model and performs
   evaluation-only forward passes on the first 50 validation cases.
9. Compares the resulting metric with the historical ~0.044026 value.

This is the final forensic step before deciding whether any further training
is necessary.

Run from project root:

python ".\\src\\segmentation_rsna_part81_original_part71_code_forensics.py"
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

SRC_DIR = PROJECT_ROOT / "src"

PART9_PATH = (
    SRC_DIR
    / "segmentation_rsna_part9_3d_dataset_loader.py"
)

PART11_PATH = (
    SRC_DIR
    / "segmentation_rsna_part11_controlled_pilot_training.py"
)

PART71_PATH = (
    SRC_DIR
    / "segmentation_rsna_part71_class_balanced_dicece_training.py"
)

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART71_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part71_class_balanced_dicece_training"
)

PART71_CKPT_DIR = (
    PART71_DIR
    / "checkpoints"
)

TRAIN_COHORT = (
    PART15_DIR
    / "part15_train_cohort.csv"
)

VAL_COHORT = (
    PART15_DIR
    / "part15_validation_cohort.csv"
)

PART15_INIT = (
    PART15_DIR
    / "checkpoints"
    / "part15_initialization_from_part11.pth"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part81_original_part71_code_forensics"
)

REPORT_DIR = OUT_DIR / "reports"


# =====================================================================
# LOCKED PROJECT VALUES
# =====================================================================

NUM_CLASSES = 6
TRAIN_N = 100
VAL_N = 50

FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)

PART71_REFERENCE_E3 = 0.04402626277260595

EXPECTED_PART15_SHA256 = (
    "0900c0e6490fdaddf455763b465d4a607acfb310349c9f9ec82a61117016aec6"
)


# =====================================================================
# GENERAL UTILITIES
# =====================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def safe_text(path: Path) -> str:

    if not path.exists():
        return ""

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def load_module(name: str, path: Path):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def relative(path: Path) -> str:

    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        )
    except Exception:
        return str(path)


def search_files(
    root: Path,
    patterns: List[str],
) -> List[Path]:

    if not root.exists():
        return []

    found = []
    seen = set()

    for pattern in patterns:

        for p in root.rglob(pattern):

            if not p.is_file():
                continue

            key = str(
                p.resolve()
            )

            if key in seen:
                continue

            seen.add(key)
            found.append(p)

    return sorted(found)


# =====================================================================
# SOURCE EXTRACTION
# =====================================================================

FORENSIC_TERMS = [
    # Loss
    "DiceCELoss",
    "DiceLoss",
    "CrossEntropyLoss",
    "cross_entropy",
    "class_weights",
    "class_weight",
    "weight=",
    "ce_weight",
    "lambda_dice",
    "lambda_ce",
    "include_background",
    "to_onehot_y",
    "softmax",
    "squared_pred",

    # Model
    "SwinUNETR",
    "feature_size",
    "in_channels",
    "out_channels",
    "spatial_dims",

    # Data
    "load_tensor_case",
    "preprocess_case",
    "center_crop",
    "centered",
    "crop",
    "R2",
    "radius",
    "pseudo",

    # Metrics
    "dice_from_prediction",
    "FGDice",
    "foreground",
    "argmax",
    "classwise",
    "Dice",

    # Training
    "epochs",
    "EPOCHS",
    "learning_rate",
    "lr",
    "weight_decay",
    "AdamW",
    "optimizer",
    "seed",
    "TRAIN_N",
    "VAL_N",
    "batch_size",

    # Checkpoint
    "torch.save",
    "torch.load",
    "model_state_dict",
    "optimizer_state_dict",
    "best_val_dice",
    "epoch",
]


def extract_context(
    text: str,
    term: str,
    radius: int = 900,
    max_hits: int = 8,
) -> List[str]:

    contexts = []

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

        contexts.append(
            text[start:end]
        )

        if len(contexts) >= max_hits:
            break

    return contexts


def ast_find_nodes(
    tree: ast.AST,
    node_type,
) -> List[ast.AST]:

    return [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            node_type,
        )
    ]


def ast_call_summary(
    tree: ast.AST,
) -> List[Dict[str, Any]]:

    output = []

    for node in ast_find_nodes(
        tree,
        ast.Call,
    ):

        try:
            function_name = ast.unparse(
                node.func
            )
        except Exception:
            function_name = "<unknown>"

        if any(
            keyword in function_name.lower()
            for keyword in [
                "loss",
                "dice",
                "cross_entropy",
                "swinunetr",
                "optimizer",
                "save",
                "load",
            ]
        ):

            kwargs = {}

            for kw in node.keywords:

                if kw.arg is None:
                    continue

                try:
                    value = ast.unparse(
                        kw.value
                    )
                except Exception:
                    value = "<unparseable>"

                kwargs[kw.arg] = value

            try:
                args = [
                    ast.unparse(arg)
                    for arg in node.args
                ]
            except Exception:
                args = []

            output.append(
                {
                    "function": function_name,
                    "args": args,
                    "kwargs": kwargs,
                    "line": getattr(
                        node,
                        "lineno",
                        None,
                    ),
                }
            )

    return output


def source_forensics() -> Dict[str, Any]:

    print(
        "\n[SOURCE FORENSICS]"
    )

    if not PART71_PATH.exists():

        return {
            "status": "PART71_SOURCE_MISSING",
            "path": str(PART71_PATH),
        }

    text = safe_text(
        PART71_PATH
    )

    result = {
        "path": str(PART71_PATH),
        "sha256": sha256_file(
            PART71_PATH
        ),
        "bytes": PART71_PATH.stat().st_size,
        "terms": {},
        "ast_calls": [],
    }

    print(
        f"  Part 71 source: {PART71_PATH}"
    )

    print(
        f"  SHA256: {result['sha256']}"
    )

    print(
        f"  Size: {result['bytes']} bytes"
    )

    for term in FORENSIC_TERMS:

        contexts = extract_context(
            text,
            term,
        )

        if contexts:

            result["terms"][term] = contexts

    try:

        tree = ast.parse(
            text
        )

        result["ast_calls"] = (
            ast_call_summary(
                tree
            )
        )

        print(
            f"  AST call records: "
            f"{len(result['ast_calls'])}"
        )

    except Exception as exc:

        result["ast_error"] = str(
            exc
        )

        print(
            f"  AST parse error: {exc}"
        )

    # Human-readable key findings.
    print(
        "\n  Key source matches:"
    )

    for term in [
        "DiceCELoss",
        "CrossEntropyLoss",
        "class_weights",
        "weight=",
        "lambda_dice",
        "lambda_ce",
        "center_crop",
        "load_tensor_case",
        "dice_from_prediction",
        "SwinUNETR",
        "AdamW",
        "torch.save",
    ]:

        count = len(
            result["terms"].get(
                term,
                [],
            )
        )

        print(
            f"    {term}: {count} context hit(s)"
        )

    return result


# =====================================================================
# SOURCE LINE EXTRACTION
# =====================================================================

def extract_relevant_source_lines() -> Dict[str, Any]:

    print(
        "\n[RELEVANT PART 71 SOURCE LINES]"
    )

    text = safe_text(
        PART71_PATH
    )

    lines = text.splitlines()

    terms = [
        "DiceCELoss",
        "CrossEntropyLoss",
        "class_weights",
        "ce_weight",
        "lambda_dice",
        "lambda_ce",
        "weight=",
        "center_crop",
        "crop",
        "load_tensor_case",
        "dice_from_prediction",
        "SwinUNETR",
        "AdamW",
        "optimizer",
        "torch.save",
        "torch.load",
        "model_state_dict",
        "validation",
        "val_dice",
        "FGDice",
    ]

    selected = []

    for i, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            term.lower()
            in line.lower()
            for term in terms
        ):

            start = max(
                1,
                i - 2,
            )

            end = min(
                len(lines),
                i + 2,
            )

            selected.append(
                {
                    "line": i,
                    "text": line,
                    "context": [
                        {
                            "line": j,
                            "text": lines[j - 1],
                        }
                        for j in range(
                            start,
                            end + 1,
                        )
                    ],
                }
            )

    print(
        f"  Relevant line hits: "
        f"{len(selected)}"
    )

    return {
        "line_count": len(lines),
        "matches": selected,
    }


# =====================================================================
# HISTORICAL REPORT FORENSICS
# =====================================================================

def historical_report_forensics() -> Dict[str, Any]:

    print(
        "\n[HISTORICAL PART 71 REPORT FORENSICS]"
    )

    files = search_files(
        PART71_DIR,
        [
            "*.txt",
            "*.json",
            "*.csv",
        ],
    )

    result = {
        "files": [],
        "reference_hits": [],
    }

    for path in files:

        text = safe_text(
            path
        )

        relative_path = relative(
            path
        )

        result["files"].append(
            {
                "path": relative_path,
                "sha256": sha256_file(
                    path
                ),
                "bytes": path.stat().st_size,
            }
        )

        # Search for exact historical number.
        if "0.044026" in text:

            result["reference_hits"].append(
                {
                    "path": relative_path,
                    "contexts": extract_context(
                        text,
                        "0.044026",
                        radius=500,
                        max_hits=10,
                    ),
                }
            )

    print(
        f"  Historical files found: "
        f"{len(files)}"
    )

    print(
        f"  Exact 0.044026 hits: "
        f"{len(result['reference_hits'])}"
    )

    return result


# =====================================================================
# CHECKPOINT DISCOVERY
# =====================================================================

def discover_part71_checkpoints() -> Dict[str, Any]:

    print(
        "\n[PART 71 CHECKPOINT DISCOVERY]"
    )

    checkpoints = search_files(
        PART71_CKPT_DIR,
        [
            "*.pth",
            "*.pt",
            "*.ckpt",
        ],
    )

    result = {
        "checkpoints": [],
    }

    for path in checkpoints:

        info = {
            "path": str(path),
            "relative_path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

        try:

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            info[
                "object_type"
            ] = type(
                checkpoint
            ).__name__

            if isinstance(
                checkpoint,
                dict,
            ):

                info[
                    "keys"
                ] = list(
                    checkpoint.keys()
                )

                info[
                    "epoch"
                ] = checkpoint.get(
                    "epoch"
                )

                info[
                    "best_val_dice"
                ] = checkpoint.get(
                    "best_val_dice"
                )

                info[
                    "history_present"
                ] = (
                    "history"
                    in checkpoint
                )

        except Exception as exc:

            info[
                "load_error"
            ] = str(
                exc
            )

        result[
            "checkpoints"
        ].append(
            info
        )

        print(
            f"  {path.name}"
        )

        if "epoch" in info:

            print(
                f"    epoch={info['epoch']} "
                f"best_val_dice="
                f"{info.get('best_val_dice')}"
            )

    return result


# =====================================================================
# CHECKPOINT SELECTION
# =====================================================================

def select_epoch3_checkpoint(
    checkpoint_info: Dict[str, Any],
) -> Optional[Path]:

    candidates = []

    for item in checkpoint_info[
        "checkpoints"
    ]:

        path = Path(
            item["path"]
        )

        name = path.name.lower()

        epoch = item.get(
            "epoch"
        )

        if epoch == 3:

            candidates.append(
                path
            )

        elif "epoch3" in name or "epoch_3" in name:

            candidates.append(
                path
            )

    # Prefer explicit epoch-3 names.
    candidates.sort(
        key=lambda p: (
            "epoch3" not in p.name.lower()
            and "epoch_3" not in p.name.lower(),
            len(p.name),
        )
    )

    if candidates:

        return candidates[0]

    return None


# =====================================================================
# CHECKPOINT STATE EXTRACTION
# =====================================================================

def extract_model_state(
    checkpoint: Any,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:

    metadata = {}

    if isinstance(
        checkpoint,
        dict,
    ):

        metadata = {
            "epoch": checkpoint.get(
                "epoch"
            ),
            "best_val_dice": checkpoint.get(
                "best_val_dice"
            ),
            "seed": checkpoint.get(
                "seed"
            ),
            "patch_size": checkpoint.get(
                "patch_size"
            ),
            "feature_size": checkpoint.get(
                "feature_size"
            ),
            "num_classes": checkpoint.get(
                "num_classes"
            ),
            "train_cases": checkpoint.get(
                "train_cases"
            ),
            "validation_cases": checkpoint.get(
                "validation_cases"
            ),
        }

        if (
            "model_state_dict"
            in checkpoint
        ):

            return (
                checkpoint[
                    "model_state_dict"
                ],
                metadata,
            )

    if isinstance(
        checkpoint,
        dict,
    ):

        # Direct state dict.
        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:

            return (
                checkpoint,
                metadata,
            )

    raise ValueError(
        "Could not extract model_state_dict"
    )


# =====================================================================
# MODEL ARCHITECTURE AUDIT
# =====================================================================

def audit_model_architecture(
    part11,
    checkpoint_path: Optional[Path],
) -> Dict[str, Any]:

    print(
        "\n[MODEL ARCHITECTURE AUDIT]"
    )

    result = {
        "checkpoint_path": (
            str(checkpoint_path)
            if checkpoint_path
            else None
        ),
    }

    try:

        device = torch.device(
            "cpu"
        )

        model = part11.create_model(
            device
        )

        result[
            "model_class"
        ] = type(
            model
        ).__name__

        result[
            "parameter_count"
        ] = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"  Model: "
            f"{result['model_class']}"
        )

        print(
            f"  Parameters: "
            f"{result['parameter_count']:,}"
        )

    except Exception as exc:

        result[
            "create_model_error"
        ] = str(
            exc
        )

        print(
            f"  Model creation failed: "
            f"{exc}"
        )

        return result

    if checkpoint_path is None:

        return result

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        state_dict, metadata = (
            extract_model_state(
                checkpoint
            )
        )

        result[
            "checkpoint_metadata"
        ] = metadata

        missing, unexpected = (
            model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        result[
            "missing_keys"
        ] = list(
            missing
        )

        result[
            "unexpected_keys"
        ] = list(
            unexpected
        )

        result[
            "state_load_exact"
        ] = (
            len(missing) == 0
            and len(unexpected) == 0
        )

        print(
            f"  State load exact: "
            f"{result['state_load_exact']}"
        )

        print(
            f"  Missing keys: "
            f"{len(missing)}"
        )

        print(
            f"  Unexpected keys: "
            f"{len(unexpected)}"
        )

    except Exception as exc:

        result[
            "checkpoint_load_error"
        ] = str(
            exc
        )

        print(
            f"  Checkpoint load failed: "
            f"{exc}"
        )

    return result


# =====================================================================
# TENSOR SHAPE / PREPROCESSING FORENSICS
# =====================================================================

def as_dhw(
    tensor: torch.Tensor,
) -> torch.Tensor:

    x = tensor.detach().cpu()

    while x.ndim > 3:

        singleton_axes = [
            i
            for i, size in enumerate(
                x.shape
            )
            if size == 1
        ]

        if not singleton_axes:

            raise ValueError(
                f"Cannot reduce "
                f"{tuple(x.shape)} to DHW"
            )

        x = x.squeeze(
            singleton_axes[0]
        )

    if x.ndim != 3:

        raise ValueError(
            f"Expected DHW, got "
            f"{tuple(x.shape)}"
        )

    return x


def center_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
):

    d, h, w = image.shape

    cd, ch, cw = CROP_SHAPE

    if (
        d < cd
        or h < ch
        or w < cw
    ):

        raise ValueError(
            f"Cannot crop {tuple(image.shape)} "
            f"to {CROP_SHAPE}"
        )

    z0 = (
        d - cd
    ) // 2

    y0 = (
        h - ch
    ) // 2

    x0 = (
        w - cw
    ) // 2

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


def mask_statistics(
    mask: torch.Tensor,
) -> Dict[str, Any]:

    mask = mask.long()

    total = int(
        mask.numel()
    )

    counts = {
        str(c): int(
            (mask == c).sum().item()
        )
        for c in range(
            NUM_CLASSES
        )
    }

    foreground = int(
        (mask > 0).sum().item()
    )

    return {
        "shape": list(
            mask.shape
        ),
        "total": total,
        "foreground": foreground,
        "foreground_fraction": (
            foreground / total
            if total
            else 0.0
        ),
        "class_counts": counts,
    }


# =====================================================================
# DIRECT EVALUATION OF HISTORICAL PART 71 CHECKPOINT
# =====================================================================

def evaluate_saved_checkpoint(
    part9,
    part11,
    checkpoint_path: Path,
) -> Dict[str, Any]:

    print(
        "\n[DIRECT EVALUATION — SAVED PART 71 EPOCH 3]"
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"  Device: {device}"
    )

    # Load model using the project's validated model factory.
    model = part11.create_model(
        device
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    state_dict, metadata = (
        extract_model_state(
            checkpoint
        )
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    val_df = pd.read_csv(
        VAL_COHORT
    ).iloc[:VAL_N].reset_index(
        drop=True
    )

    per_case = []

    total_target_fg = 0
    total_pred_fg = 0

    # Classwise accumulators.
    intersection = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    pred_count = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    target_count = np.zeros(
        NUM_CLASSES,
        dtype=np.float64,
    )

    for i, (_, row) in enumerate(
        val_df.iterrows()
    ):

        loaded = part11.load_tensor_case(
            row,
            part9,
        )

        image = as_dhw(
            loaded[0]
        ).float()

        target = as_dhw(
            loaded[1]
        ).long()

        image, target = center_crop(
            image,
            target,
        )

        # [1, 1, D, H, W]
        input_tensor = (
            image.unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        target_tensor = (
            target.unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            prediction = torch.argmax(
                logits,
                dim=1,
            ).squeeze(0).cpu()

        # Ensure prediction shape is exactly target shape.
        if prediction.shape != target.shape:

            prediction = F.interpolate(
                prediction.float()
                .unsqueeze(0)
                .unsqueeze(0),
                size=target.shape,
                mode="nearest",
            ).squeeze().long()

        case_target_fg = int(
            (target > 0).sum().item()
        )

        case_pred_fg = int(
            (prediction > 0).sum().item()
        )

        total_target_fg += (
            case_target_fg
        )

        total_pred_fg += (
            case_pred_fg
        )

        for c in range(
            NUM_CLASSES
        ):

            p = (
                prediction == c
            )

            t = (
                target == c
            )

            inter = int(
                (p & t).sum().item()
            )

            intersection[c] += inter

            pred_count[c] += int(
                p.sum().item()
            )

            target_count[c] += int(
                t.sum().item()
            )

        # Foreground Dice for this case.
        p_fg = (
            prediction > 0
        )

        t_fg = (
            target > 0
        )

        inter_fg = int(
            (p_fg & t_fg).sum().item()
        )

        denom_fg = (
            int(p_fg.sum().item())
            + int(t_fg.sum().item())
        )

        case_fg_dice = (
            2.0 * inter_fg / denom_fg
            if denom_fg
            else 1.0
        )

        per_case.append(
            {
                "index": i,
                "target_fg": case_target_fg,
                "pred_fg": case_pred_fg,
                "fg_dice": case_fg_dice,
            }
        )

        if (
            (i + 1) % 10 == 0
            or i == len(val_df) - 1
        ):

            print(
                f"  Evaluated "
                f"{i + 1}/{len(val_df)}"
            )

    # -----------------------------------------------------------------
    # Aggregated foreground Dice
    # -----------------------------------------------------------------

    global_fg_intersection = (
        intersection[1:].sum()
    )

    global_fg_pred = (
        pred_count[1:].sum()
    )

    global_fg_target = (
        target_count[1:].sum()
    )

    global_fg_dice = (
        2.0
        * global_fg_intersection
        / (
            global_fg_pred
            + global_fg_target
        )
        if (
            global_fg_pred
            + global_fg_target
        )
        else 1.0
    )

    # Macro classwise foreground Dice.
    classwise = {}

    for c in range(
        1,
        NUM_CLASSES,
    ):

        denom = (
            pred_count[c]
            + target_count[c]
        )

        dice = (
            2.0
            * intersection[c]
            / denom
            if denom
            else 1.0
        )

        classwise[str(c)] = {
            "dice": float(dice),
            "intersection": float(
                intersection[c]
            ),
            "pred_voxels": float(
                pred_count[c]
            ),
            "target_voxels": float(
                target_count[c]
            ),
        }

    mean_case_fg_dice = float(
        np.mean(
            [
                x["fg_dice"]
                for x in per_case
            ]
        )
    )

    result = {
        "device": str(device),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_metadata": metadata,
        "n_validation_cases": len(
            per_case
        ),
        "total_target_foreground": int(
            total_target_fg
        ),
        "total_predicted_foreground": int(
            total_pred_fg
        ),
        "global_foreground_dice": float(
            global_fg_dice
        ),
        "mean_case_foreground_dice": mean_case_fg_dice,
        "classwise_foreground": classwise,
        "per_case": per_case,
    }

    print(
        "\n  DIRECT CHECKPOINT METRICS"
    )

    print(
        f"    Global foreground Dice: "
        f"{global_fg_dice:.9f}"
    )

    print(
        f"    Mean case foreground Dice: "
        f"{mean_case_fg_dice:.9f}"
    )

    print(
        f"    Total target FG: "
        f"{total_target_fg}"
    )

    print(
        f"    Total predicted FG: "
        f"{total_pred_fg}"
    )

    print(
        "\n  Classwise Dice:"
    )

    for c, values in classwise.items():

        print(
            f"    C{c}: "
            f"{values['dice']:.9f} "
            f"(pred={values['pred_voxels']:.0f}, "
            f"target={values['target_voxels']:.0f})"
        )

    return result


# =====================================================================
# METRIC DEFINITION FORENSICS
# =====================================================================

def metric_forensics(
    part11,
    direct_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    print(
        "\n[METRIC FORENSICS]"
    )

    result = {
        "part11_metric_functions": {},
        "direct_evaluation": None,
    }

    for name in [
        "dice_from_prediction",
        "dice_score",
        "compute_dice",
        "foreground_dice",
        "evaluate",
    ]:

        if hasattr(
            part11,
            name,
        ):

            obj = getattr(
                part11,
                name,
            )

            try:

                signature = str(
                    inspect.signature(
                        obj
                    )
                )

            except Exception:

                signature = "<signature unavailable>"

            result[
                "part11_metric_functions"
            ][name] = {
                "signature": signature,
                "type": type(obj).__name__,
            }

            print(
                f"  {name}{signature}"
            )

    if direct_eval is not None:

        result[
            "direct_evaluation"
        ] = {
            "global_foreground_dice":
                direct_eval[
                    "global_foreground_dice"
                ],
            "mean_case_foreground_dice":
                direct_eval[
                    "mean_case_foreground_dice"
                ],
        }

    return result


# =====================================================================
# DECISION LOGIC
# =====================================================================

def determine_status(
    source: Dict[str, Any],
    report: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    direct_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    findings = []

    # ---------------------------------------------------------------
    # Historical reference exists?
    # ---------------------------------------------------------------

    exact_reference_found = (
        len(
            report[
                "reference_hits"
            ]
        )
        > 0
    )

    if exact_reference_found:

        findings.append(
            "HISTORICAL_PART71_REFERENCE_0_044026_CONFIRMED"
        )

    # ---------------------------------------------------------------
    # Epoch-3 checkpoint available?
    # ---------------------------------------------------------------

    epoch3_available = (
        direct_eval is not None
    )

    if epoch3_available:

        global_dice = (
            direct_eval[
                "global_foreground_dice"
            ]
        )

        mean_case_dice = (
            direct_eval[
                "mean_case_foreground_dice"
            ]
        )

        global_diff = abs(
            global_dice
            - PART71_REFERENCE_E3
        )

        mean_diff = abs(
            mean_case_dice
            - PART71_REFERENCE_E3
        )

        if global_diff <= 0.005:

            findings.append(
                "SAVED_PART71_EPOCH3_GLOBAL_DICE_MATCHES_HISTORICAL_REFERENCE"
            )

        else:

            findings.append(
                "SAVED_PART71_EPOCH3_GLOBAL_DICE_DOES_NOT_MATCH_HISTORICAL_REFERENCE"
            )

        if mean_diff <= 0.005:

            findings.append(
                "SAVED_PART71_EPOCH3_MEAN_CASE_DICE_MATCHES_HISTORICAL_REFERENCE"
            )

        else:

            findings.append(
                "SAVED_PART71_EPOCH3_MEAN_CASE_DICE_DOES_NOT_MATCH_HISTORICAL_REFERENCE"
            )

        if (
            global_diff > 0.005
            and mean_diff > 0.005
        ):

            status = (
                "PART71_CHECKPOINT_ITSELF_DOES_NOT_REPRODUCE_REPORTED_0_044026"
            )

        else:

            status = (
                "PART71_CHECKPOINT_CAN_EXPLAIN_REPORTED_REFERENCE_WITH_METRIC_DIFFERENCE"
            )

    else:

        status = (
            "PART71_EPOCH3_CHECKPOINT_NOT_AVAILABLE_FOR_DIRECT_FORENSIC_EVALUATION"
        )

    # ---------------------------------------------------------------
    # Architecture metadata
    # ---------------------------------------------------------------

    return {
        "status": status,
        "findings": findings,
        "exact_historical_reference_found":
            exact_reference_found,
        "epoch3_checkpoint_available":
            epoch3_available,
        "recommendation": (
            "Do not start another training ablation. "
            "If the saved Part 71 checkpoint matches 0.044026, "
            "lock the original Part 71 implementation/metric as the "
            "historical result. If it does not match, investigate the "
            "historical metric/report generation rather than training "
            "again."
        ),
    }


# =====================================================================
# REPORT WRITER
# =====================================================================

def write_reports(
    report: Dict[str, Any],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        REPORT_DIR
        / "part81_forensic_summary.json"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    decision = report[
        "decision"
    ]

    direct = report.get(
        "direct_evaluation"
    )

    row = {
        "status": decision[
            "status"
        ],
        "historical_reference": PART71_REFERENCE_E3,
        "epoch3_checkpoint_found":
            decision[
                "epoch3_checkpoint_available"
            ],
        "direct_global_fg_dice":
            (
                direct[
                    "global_foreground_dice"
                ]
                if direct
                else None
            ),
        "direct_mean_case_fg_dice":
            (
                direct[
                    "mean_case_foreground_dice"
                ]
                if direct
                else None
            ),
        "predicted_foreground":
            (
                direct[
                    "total_predicted_foreground"
                ]
                if direct
                else None
            ),
        "target_foreground":
            (
                direct[
                    "total_target_foreground"
                ]
                if direct
                else None
            ),
    }

    csv_path = (
        REPORT_DIR
        / "part81_forensic_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(row)

    txt = []

    txt.append(
        "PART 81 — ORIGINAL PART 71 CODE FORENSICS"
    )

    txt.append(
        "=" * 72
    )

    txt.append("")

    txt.append(
        f"STATUS: {decision['status']}"
    )

    txt.append("")

    txt.append(
        "This was an audit-only operation."
    )

    txt.append(
        "No optimizer.step() was executed."
    )

    txt.append(
        "No backward() was executed."
    )

    txt.append(
        "No checkpoint was modified."
    )

    txt.append("")

    txt.append(
        "HISTORICAL REFERENCE"
    )

    txt.append(
        f"Part 71 E3 reference: "
        f"{PART71_REFERENCE_E3:.12f}"
    )

    txt.append("")

    txt.append(
        "DIRECT CHECKPOINT EVALUATION"
    )

    if direct:

        txt.append(
            f"Global foreground Dice: "
            f"{direct['global_foreground_dice']:.12f}"
        )

        txt.append(
            f"Mean case foreground Dice: "
            f"{direct['mean_case_foreground_dice']:.12f}"
        )

        txt.append(
            f"Predicted foreground voxels: "
            f"{direct['total_predicted_foreground']}"
        )

        txt.append(
            f"Target foreground voxels: "
            f"{direct['total_target_foreground']}"
        )

    else:

        txt.append(
            "Direct evaluation unavailable."
        )

    txt.append("")

    txt.append(
        "FINDINGS"
    )

    for finding in decision[
        "findings"
    ]:

        txt.append(
            f"- {finding}"
        )

    txt.append("")

    txt.append(
        "RECOMMENDATION"
    )

    txt.append(
        decision[
            "recommendation"
        ]
    )

    txt.append("")

    txt.append(
        "SOURCE FILE"
    )

    txt.append(
        report[
            "source"
        ]["path"]
    )

    txt.append(
        f"SHA256: "
        f"{report['source']['sha256']}"
    )

    (
        REPORT_DIR
        / "part81_report.txt"
    ).write_text(
        "\n".join(txt),
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 78
    )

    print(
        "PART 81 — ORIGINAL PART 71 CODE FORENSICS"
    )

    print(
        "=" * 78
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Basic path validation
    # ---------------------------------------------------------------

    required = [
        PART9_PATH,
        PART11_PATH,
        PART71_PATH,
        TRAIN_COHORT,
        VAL_COHORT,
        PART15_INIT,
    ]

    print(
        "\n[REQUIRED FILE CHECK]"
    )

    for path in required:

        print(
            f"  {'FOUND' if path.exists() else 'MISSING'} "
            f"{relative(path)}"
        )

    if not PART71_PATH.exists():

        raise FileNotFoundError(
            f"Original Part 71 source not found:\n"
            f"{PART71_PATH}"
        )

    if not VAL_COHORT.exists():

        raise FileNotFoundError(
            f"Validation cohort not found:\n"
            f"{VAL_COHORT}"
        )

    # ---------------------------------------------------------------
    # Verify Part 15 checkpoint identity.
    # ---------------------------------------------------------------

    print(
        "\n[PART 15 INITIALIZATION IDENTITY]"
    )

    part15_hash = sha256_file(
        PART15_INIT
    )

    print(
        f"  Actual SHA256: {part15_hash}"
    )

    print(
        f"  Expected SHA256: "
        f"{EXPECTED_PART15_SHA256}"
    )

    print(
        f"  Match: "
        f"{part15_hash == EXPECTED_PART15_SHA256}"
    )

    # ---------------------------------------------------------------
    # Source forensics.
    # ---------------------------------------------------------------

    source = source_forensics()

    source_lines = (
        extract_relevant_source_lines()
    )

    # ---------------------------------------------------------------
    # Historical report forensics.
    # ---------------------------------------------------------------

    historical_report = (
        historical_report_forensics()
    )

    # ---------------------------------------------------------------
    # Discover saved Part 71 checkpoints.
    # ---------------------------------------------------------------

    checkpoint_info = (
        discover_part71_checkpoints()
    )

    epoch3_checkpoint = (
        select_epoch3_checkpoint(
            checkpoint_info
        )
    )

    if epoch3_checkpoint:

        print(
            f"\n  Selected epoch-3 checkpoint:"
        )

        print(
            f"    {epoch3_checkpoint}"
        )

    else:

        print(
            "\n  No epoch-3 checkpoint selected."
        )

    # ---------------------------------------------------------------
    # Load validated project modules.
    # ---------------------------------------------------------------

    part9 = load_module(
        "part9_part81",
        PART9_PATH,
    )

    part11 = load_module(
        "part11_part81",
        PART11_PATH,
    )

    # ---------------------------------------------------------------
    # Architecture audit.
    # ---------------------------------------------------------------

    architecture = (
        audit_model_architecture(
            part11,
            epoch3_checkpoint,
        )
    )

    # ---------------------------------------------------------------
    # Metric API audit.
    # ---------------------------------------------------------------

    direct_evaluation = None

    if epoch3_checkpoint:

        try:

            direct_evaluation = (
                evaluate_saved_checkpoint(
                    part9,
                    part11,
                    epoch3_checkpoint,
                )
            )

        except Exception as exc:

            print(
                "\n[DIRECT EVALUATION ERROR]"
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )

            direct_evaluation = {
                "error": str(exc),
                "error_type":
                    type(exc).__name__,
            }

    else:

        print(
            "\n[DIRECT EVALUATION SKIPPED]"
        )

    metric = metric_forensics(
        part11,
        (
            direct_evaluation
            if direct_evaluation
            and "error"
            not in direct_evaluation
            else None
        ),
    )

    # ---------------------------------------------------------------
    # Final decision.
    # ---------------------------------------------------------------

    decision = determine_status(
        source=source,
        report=historical_report,
        checkpoint_info=checkpoint_info,
        direct_eval=(
            direct_evaluation
            if direct_evaluation
            and "error"
            not in direct_evaluation
            else None
        ),
    )

    report = {
        "part": 81,
        "title":
            "Original Part 71 Code Forensics",
        "audit_only": True,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "checkpoint_modifications": 0,
        "part15_sha256": part15_hash,
        "part15_sha256_matches":
            (
                part15_hash
                == EXPECTED_PART15_SHA256
            ),
        "source": source,
        "source_relevant_lines":
            source_lines,
        "historical_report":
            historical_report,
        "checkpoint_discovery":
            checkpoint_info,
        "selected_epoch3_checkpoint":
            (
                str(epoch3_checkpoint)
                if epoch3_checkpoint
                else None
            ),
        "architecture":
            architecture,
        "direct_evaluation":
            direct_evaluation,
        "metric_forensics":
            metric,
        "decision":
            decision,
    }

    write_reports(
        report
    )

    # ---------------------------------------------------------------
    # Final console summary.
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        "PART 81 FORENSICS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        f"Status: {decision['status']}"
    )

    print(
        f"Historical Part 71 E3 reference: "
        f"{PART71_REFERENCE_E3:.9f}"
    )

    if (
        direct_evaluation
        and "error"
        not in direct_evaluation
    ):

        print(
            f"Saved Part 71 epoch-3 global FG Dice: "
            f"{direct_evaluation['global_foreground_dice']:.9f}"
        )

        print(
            f"Saved Part 71 epoch-3 mean-case FG Dice: "
            f"{direct_evaluation['mean_case_foreground_dice']:.9f}"
        )

        print(
            f"Saved checkpoint predicted FG: "
            f"{direct_evaluation['total_predicted_foreground']}"
        )

        print(
            f"Saved checkpoint target FG: "
            f"{direct_evaluation['total_target_foreground']}"
        )

    elif (
        direct_evaluation
        and "error"
        in direct_evaluation
    ):

        print(
            f"Direct evaluation error: "
            f"{direct_evaluation['error']}"
        )

    else:

        print(
            "Saved epoch-3 checkpoint evaluation: unavailable"
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
        REPORT_DIR.glob(
            "part81_*"
        )
    ):

        print(
            f"  {path}"
        )

    print(
        "\nNO TRAINING WAS PERFORMED."
    )


if __name__ == "__main__":
    main()
