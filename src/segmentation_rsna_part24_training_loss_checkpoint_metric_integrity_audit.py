"""
PHASE 4 - PART 24
RSNA-ONLY TRAINING LOSS / CHECKPOINT METRIC INTEGRITY AUDIT

Evaluation / audit only.
No training is performed.
No model weights are modified.
No optimizer is created.
SPIDER is not used.
RSNA test set is not used.

Purpose:
Investigate the contradiction established in Part 23:
    Part 15 checkpoint metadata: best_val_dice ~= 0.668
    Explicit foreground Dice from the same checkpoint: 0.0
    Predictions: foreground-empty

This audit:
  1. Inspects Part 15 source for loss/metric configuration.
  2. Inspects Part 15 training history and checkpoint metadata.
  3. Reconstructs the exact Part 15 validation cohort.
  4. Loads the exact Part 15 checkpoint.
  5. Computes DiceCELoss directly on exact validation cases.
  6. Computes explicit foreground Dice and class-wise Dice.
  7. Computes both foreground-excluding-background and all-class Dice.
  8. Tests whether a background-inclusive Dice can explain ~0.668.
  9. Reports channel/target compatibility.
 10. Does NOT modify the checkpoint or train anything.

Designed for:
Windows + Python venv + PyTorch + MONAI + the existing project structure.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from monai.losses import DiceCELoss
except Exception as exc:
    print("\nERROR: MONAI DiceCELoss could not be imported.")
    print(repr(exc))
    raise


# ============================================================================
# PROJECT PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RSNA_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

PART11_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "segmentation_rsna_part11_controlled_pilot_training_corrected.py"
)

PART15_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "src" / "segmentation_rsna_part15_extended_controlled_training.py",
    PROJECT_ROOT / "src" / "segmentation_rsna_part15_extended_controlled_training_corrected.py",
    PROJECT_ROOT / "src" / "segmentation_rsna_part15_extended_controlled_training_v2.py",
]

PART15_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

PART15_CHECKPOINT = PART15_DIR / "checkpoints" / "best_model.pth"
PART15_HISTORY = PART15_DIR / "part15_training_history.csv"
PART15_TRAIN_COHORT = PART15_DIR / "part15_train_cohort.csv"
PART15_VAL_COHORT = PART15_DIR / "part15_validation_cohort.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part24_training_loss_checkpoint_metric_integrity_audit"
)
REPORT_DIR = OUTPUT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (64, 96, 96)
NUM_CLASSES = 6
BACKGROUND_CLASS = 0
MAX_DIAGNOSTIC_CASES = 10

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================================
# UTILITIES
# ============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def import_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_part15_source() -> Optional[Path]:
    for p in PART15_SOURCE_CANDIDATES:
        if p.exists():
            return p

    # Conservative fallback: only names beginning with segmentation_rsna_part15.
    matches = sorted(SCRIPT_DIR.glob("segmentation_rsna_part15*.py"))
    return matches[0] if matches else None


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    y = x.detach()
    return {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
        "numel": int(y.numel()),
        "finite": bool(torch.isfinite(y).all().item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.float().mean().item()),
    }


def normalize_image_mask(
    image: Any,
    mask: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize the known Part 11 contract to:
      image -> [1,1,D,H,W]
      mask  -> [1,D,H,W]
    """
    image_t = image if torch.is_tensor(image) else torch.as_tensor(image)
    mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask)

    image_t = image_t.float()
    mask_t = mask_t.long()

    # Image contract.
    if image_t.ndim == 3:
        image_t = image_t.unsqueeze(0).unsqueeze(0)
    elif image_t.ndim == 4:
        # Expected [C,D,H,W] where C=1.
        image_t = image_t.unsqueeze(0)
    elif image_t.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected image dimensions: {tuple(image_t.shape)}")

    # Mask contract.
    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(0)
    elif mask_t.ndim == 4:
        pass
    elif mask_t.ndim == 5 and mask_t.shape[1] == 1:
        mask_t = mask_t[:, 0]
    else:
        raise ValueError(f"Unexpected mask dimensions: {tuple(mask_t.shape)}")

    return image_t, mask_t


def one_hot_target(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    [B,D,H,W] -> [B,C,D,H,W]
    """
    if target.ndim != 4:
        raise ValueError(f"Expected [B,D,H,W], got {tuple(target.shape)}")

    if int(target.min()) < 0 or int(target.max()) >= num_classes:
        raise ValueError(
            f"Target contains labels outside [0,{num_classes-1}]: "
            f"min={int(target.min())}, max={int(target.max())}"
        )

    return torch.nn.functional.one_hot(
        target.long(), num_classes=num_classes
    ).permute(0, 4, 1, 2, 3).float()


def explicit_class_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool,
    smooth: float = 0.0,
) -> List[float]:
    """
    Hard-label Dice computed directly from argmax prediction and target.
    Empty/empty is treated as 1.0 for a class.
    """
    pred = prediction.detach().long()
    tgt = target.detach().long()

    start = 0 if include_background else 1
    scores = []

    for c in range(start, num_classes):
        p = pred == c
        t = tgt == c
        inter = torch.logical_and(p, t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()

        if p_sum == 0 and t_sum == 0:
            dice = 1.0
        elif p_sum + t_sum == 0:
            dice = 0.0
        else:
            dice = (2.0 * inter + smooth) / (p_sum + t_sum + smooth)

        scores.append(float(dice))

    return scores


def mean_dice_from_scores(scores: List[float]) -> float:
    return float(np.mean(scores)) if scores else float("nan")


def inspect_source_text(path: Optional[Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path) if path else None,
        "found": bool(path and path.exists()),
        "matches": {},
        "parse_ok": False,
    }

    if not path or not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")

    patterns = {
        "DiceCELoss": r"DiceCELoss\s*\((.*?)\)",
        "dice_from_prediction": r"dice_from_prediction",
        "best_val_dice": r"best_val_dice",
        "val_dice": r"val_dice",
        "argmax": r"argmax\s*\(",
        "softmax": r"softmax\s*\(",
        "include_background": r"include_background",
        "to_onehot_y": r"to_onehot_y",
        "softmax_arg": r"softmax\s*=\s*True",
        "sigmoid": r"sigmoid\s*\(",
        "num_classes": r"num_classes",
        "loss_fn": r"loss_fn",
    }

    for name, pattern in patterns.items():
        hits = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        snippets = []
        for m in hits[:8]:
            a = max(0, m.start() - 180)
            b = min(len(text), m.end() + 280)
            snippets.append(text[a:b].replace("\n", " "))
        result["matches"][name] = snippets

    try:
        ast.parse(text)
        result["parse_ok"] = True
    except Exception as exc:
        result["parse_error"] = repr(exc)

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths() -> Tuple[Path, Optional[Path]]:
    section("PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "Part 11 source": PART11_SOURCE,
        "Part 15 checkpoint": PART15_CHECKPOINT,
        "Part 15 history": PART15_HISTORY,
        "Part 15 train cohort": PART15_TRAIN_COHORT,
        "Part 15 validation cohort": PART15_VAL_COHORT,
    }

    missing = []
    for name, path in required.items():
        state = "FOUND" if path.exists() else "MISSING"
        print(f"{name:<36}: {state}")
        if not path.exists():
            missing.append(name)

    part15_source = find_part15_source()
    print(
        f"{'Part 15 source':<36}: "
        f"{'FOUND -> ' + str(part15_source) if part15_source else 'NOT FOUND'}"
    )

    if missing:
        raise FileNotFoundError(
            "Missing required Part 24 input(s):\n" + "\n".join(missing)
        )

    return PART15_CHECKPOINT, part15_source


# ============================================================================
# PART 11
# ============================================================================

def import_part11():
    section("IMPORTING VALIDATED PART 11")

    module = import_module_from_file("part11_part24", PART11_SOURCE)

    print("âœ“ Corrected Part 11 imported.")

    for name in [
        "create_model",
        "preprocess_case",
        "load_tensor_case",
        "select_pilot_rows",
    ]:
        if hasattr(module, name):
            try:
                print(f"{name:<36}: {inspect.signature(getattr(module, name))}")
            except Exception:
                print(f"{name:<36}: available")

    return module


# ============================================================================
# CHECKPOINT / HISTORY
# ============================================================================

def load_checkpoint(path: Path) -> Dict[str, Any]:
    section("LOADING PART 15 BEST CHECKPOINT")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    print(f"Checkpoint SHA256                    : {sha256_file(path)}")
    print(f"Checkpoint keys                      : {list(checkpoint.keys())}")

    for key in [
        "epoch",
        "best_val_dice",
        "seed",
        "feature_size",
        "num_classes",
        "train_cases",
        "validation_cases",
        "patch_size",
        "source_checkpoint",
    ]:
        if key in checkpoint:
            print(f"{key:<36}: {checkpoint[key]}")

    return checkpoint


def load_history() -> pd.DataFrame:
    section("LOADING PART 15 TRAINING HISTORY")

    df = pd.read_csv(PART15_HISTORY)
    print(f"History rows                         : {len(df)}")
    print(f"History columns                      : {list(df.columns)}")
    print("\nHistory:")
    print(df.to_string(index=False))

    return df


# ============================================================================
# COHORT
# ============================================================================

def load_validation_cohort() -> pd.DataFrame:
    section("LOADING EXACT PART 15 VALIDATION COHORT")

    df = pd.read_csv(PART15_VAL_COHORT)

    print(f"Validation cohort rows               : {len(df)}")
    print(f"Columns                               : {list(df.columns)}")

    if len(df) == 0:
        raise ValueError("Part 15 validation cohort is empty.")

    return df


# ============================================================================
# MODEL
# ============================================================================

def create_model(part11, device: torch.device) -> torch.nn.Module:
    section("CREATING SWIN-UNETR")

    factory = getattr(part11, "create_model")
    model = factory(device)

    print(f"Total parameters                     : {sum(p.numel() for p in model.parameters())}")

    return model


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> None:
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Missing keys                         : {len(missing)}")
    print(f"Unexpected keys                      : {len(unexpected)}")

    if missing:
        print("Missing:", missing)
    if unexpected:
        print("Unexpected:", unexpected)

    model.to(device)
    model.eval()

    print("âœ“ Checkpoint loaded successfully.")


# ============================================================================
# LOSS CONFIGURATION
# ============================================================================

def build_loss_variants() -> Dict[str, torch.nn.Module]:
    """
    Construct explicit loss variants so the audit can identify which
    interpretation is consistent with the recorded Part 15 value.
    """
    return {
        "dice_ce_default": DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        ),
        "dice_ce_no_background": DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        ),
        "dice_only_default": DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=0.0,
        ),
        "ce_only": DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.0,
            lambda_ce=1.0,
        ),
    }


# ============================================================================
# CASE AUDIT
# ============================================================================

@torch.no_grad()
def audit_case(
    part11,
    part9,
    model,
    row: pd.Series,
    device: torch.device,
    losses: Dict[str, torch.nn.Module],
) -> Dict[str, Any]:

    if part9 is None:
        raise RuntimeError("Internal error: part9 loader is None.")

    image, mask, info = part11.load_tensor_case(row, part9)

    image, mask = normalize_image_mask(image, mask)
    image = image.to(device)
    mask = mask.to(device)

    logits = model(image)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,D,H,W], got {tuple(logits.shape)}")

    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} output classes, got {logits.shape[1]}"
        )

    target = mask

    # Hard prediction.
    pred = torch.argmax(logits, dim=1)

    # Probabilities.
    probs = torch.softmax(logits.float(), dim=1)

    # Explicit hard-label metrics.
    fg_scores = explicit_class_dice(
        pred, target, NUM_CLASSES, include_background=False
    )
    all_scores = explicit_class_dice(
        pred, target, NUM_CLASSES, include_background=True
    )

    result: Dict[str, Any] = {
        "study_id": str(row.get("study_id", "")),
        "series_id": str(row.get("series_id", "")),
        "series_description": str(row.get("series_description", "")),
        "losses": {},
        "foreground_dice": mean_dice_from_scores(fg_scores),
        "all_class_dice": mean_dice_from_scores(all_scores),
        "background_dice": all_scores[0],
        "pred_foreground_voxels": int((pred != BACKGROUND_CLASS).sum().item()),
        "target_foreground_voxels": int((target != BACKGROUND_CLASS).sum().item()),
        "pred_label_counts": {
            str(c): int((pred == c).sum().item()) for c in range(NUM_CLASSES)
        },
        "target_label_counts": {
            str(c): int((target == c).sum().item()) for c in range(NUM_CLASSES)
        },
        "logit_mean": [
            float(logits[:, c].float().mean().item())
            for c in range(NUM_CLASSES)
        ],
        "prob_mean": [
            float(probs[:, c].mean().item())
            for c in range(NUM_CLASSES)
        ],
        "max_foreground_probability": float(
            probs[:, 1:].max().item()
        ),
        "mean_background_probability": float(
            probs[:, 0].mean().item()
        ),
        "loader_info": safe_json(info),
    }

    for name, loss_fn in losses.items():
        try:
            result["losses"][name] = float(loss_fn(logits, target.unsqueeze(1)).item())
        except Exception as exc:
            result["losses"][name] = f"ERROR: {type(exc).__name__}: {exc}"

    for c in range(1, NUM_CLASSES):
        result[f"dice_class_{c}"] = fg_scores[c - 1]

    result["dice_background"] = all_scores[0]

    return result


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    section("PHASE 4 - PART 24")
    print("RSNA-ONLY TRAINING LOSS / CHECKPOINT METRIC INTEGRITY AUDIT")
    print()
    print("Evaluation / audit only.")
    print("No training is performed.")
    print("No model weights are modified.")
    print("No optimizer is created.")
    print("SPIDER is not used.")
    print("RSNA test set is not used.")
    print()
    print("Purpose:")
    print("Investigate the Part 23 contradiction between the recorded")
    print("Part 15 Dice (~0.668) and explicit foreground Dice (=0.0).")

    checkpoint_path, part15_source = validate_paths()

    section("PYTORCH / GPU ENVIRONMENT")
    print(f"PyTorch version                       : {torch.__version__}")
    print(f"CUDA available                        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device                                : {device}")
        print(f"GPU                                   : {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory                            : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print(f"Device                                : {device}")
        print("WARNING: CUDA unavailable; audit will run on CPU.")

    print(f"Patch size                            : {PATCH_SIZE}")
    print(f"Classes                               : {NUM_CLASSES}")

    part11 = import_part11()

    section("INSPECTING PART 15 TRAINING SOURCE")
    source_audit = inspect_source_text(part15_source)

    if source_audit["found"]:
        print(f"Part 15 source                         : {part15_source}")
        print(f"AST parse successful                   : {source_audit['parse_ok']}")

        for key, snippets in source_audit["matches"].items():
            if snippets:
                print(f"\n--- {key} ({len(snippets)} matches shown) ---")
                for s in snippets[:3]:
                    print(" ", s[:700])
    else:
        print("Part 15 source file was not found.")
        print("The checkpoint/history audit will continue.")

    history = load_history()
    val_cohort = load_validation_cohort()
    checkpoint = load_checkpoint(checkpoint_path)

    section("RECONSTRUCTING MODEL")
    model = create_model(part11, device)
    load_model_weights(model, checkpoint, device)

    section("EXPLICIT LOSS CONFIGURATION")
    losses = build_loss_variants()
    for name, loss_fn in losses.items():
        print(f"{name:<36}: {loss_fn}")

    section("LOADING PART 9")
    try:
        # The project may use a differently named Part 9 source file.
        # First use Part 11's explicit loader factory, if present.
        if hasattr(part11, "load_part9_module"):
            part9 = part11.load_part9_module()
            print("âœ“ Part 9 loader created through Part 11.")
        else:
            candidates = [
                PROJECT_ROOT / "src" / "segmentation_rsna_part9_dataloader.py",
                PROJECT_ROOT / "src" / "segmentation_rsna_part9_dataset_construction.py",
                PROJECT_ROOT / "src" / "segmentation_rsna_part9_dataset_loader.py",
            ]
            candidates += sorted(SCRIPT_DIR.glob("*part9*.py"))
            candidates += sorted((PROJECT_ROOT / "src").glob("**/*part9*.py"))

            part9_source = next((x for x in candidates if x.exists()), None)

            if part9_source is None:
                raise FileNotFoundError(
                    "Could not locate a standalone Part 9 Python source."
                )

            part9 = import_module_from_file("part9_part24", part9_source)
            print(f"âœ“ Part 9 loader imported: {part9_source}")
    except Exception as exc:
        print(f"Part 9 loader creation failed: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "Part 24 requires the same Part 9 loader contract used by Part 11. "
            "Could not construct it."
        ) from exc

    section("RUNNING CHECKPOINT / METRIC INTEGRITY AUDIT")

    rows = []
    errors = []

    n = min(MAX_DIAGNOSTIC_CASES, len(val_cohort))

    for i in range(n):
        row = val_cohort.iloc[i]

        try:
            result = audit_case(
                part11=part11,
                part9=part9,
                model=model,
                row=row,
                device=device,
                losses=losses,
            )
            rows.append(result)

            print(
                f"[{i+1:03d}/{n:03d}] "
                f"{result['study_id']} | {result['series_id']} | "
                f"FG_Dice={result['foreground_dice']:.6f} | "
                f"AllDice={result['all_class_dice']:.6f} | "
                f"BG_Dice={result['background_dice']:.6f} | "
                f"PredFG={result['pred_foreground_voxels']} | "
                f"TargetFG={result['target_foreground_voxels']}"
            )

            print(
                f"           "
                f"DiceCE(default)={result['losses']['dice_ce_default']} | "
                f"DiceCE(no-bg)={result['losses']['dice_ce_no_background']}"
            )

        except Exception as exc:
            error = {
                "index": i,
                "study_id": str(row.get("study_id", "")),
                "series_id": str(row.get("series_id", "")),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error)

            print(
                f"[{i+1:03d}/{n:03d}] ERROR "
                f"{error['study_id']} | {error['series_id']} | "
                f"{error['error_type']}: {error['error']}"
            )

    if not rows:
        raise RuntimeError("No validation cases completed successfully.")

    df = pd.DataFrame(rows)

    section("PART 24 AGGREGATED METRICS")

    numeric = [
        "foreground_dice",
        "all_class_dice",
        "background_dice",
        "pred_foreground_voxels",
        "target_foreground_voxels",
    ]

    for col in numeric:
        if col in df:
            print(f"{col:<36}: {df[col].mean()}")

    print(
        f"Empty foreground predictions             : "
        f"{int((df['pred_foreground_voxels'] == 0).sum())}/{len(df)}"
    )

    print(
        f"Mean explicit foreground Dice            : "
        f"{df['foreground_dice'].mean():.6f}"
    )
    print(
        f"Mean explicit all-class Dice             : "
        f"{df['all_class_dice'].mean():.6f}"
    )
    print(
        f"Mean background Dice                     : "
        f"{df['background_dice'].mean():.6f}"
    )

    for name in losses:
        vals = []
        for x in df["losses"]:
            pass

    # Expand loss dictionary columns.
    for name in losses:
        df[f"loss_{name}"] = df["losses"].apply(
            lambda d: d.get(name) if isinstance(d, dict) else None
        )

    section("CLASS-WISE DICE")

    for c in range(1, NUM_CLASSES):
        col = f"dice_class_{c}"
        print(
            f"{c}: {CLASS_NAMES[c]:35} "
            f"Dice={df[col].mean():.6f} "
            f"Zero={(df[col] == 0).sum()}/{len(df)}"
        )

    section("RECORDED PART 15 METRIC COMPARISON")

    recorded = checkpoint.get("best_val_dice", np.nan)
    explicit_fg = float(df["foreground_dice"].mean())
    explicit_all = float(df["all_class_dice"].mean())
    bg = float(df["background_dice"].mean())

    print(f"Part 15 checkpoint best_val_dice        : {recorded}")
    print(f"Part 24 explicit foreground Dice        : {explicit_fg:.6f}")
    print(f"Part 24 explicit all-class Dice         : {explicit_all:.6f}")
    print(f"Part 24 background Dice                 : {bg:.6f}")

    if math.isfinite(float(recorded)):
        print(f"Difference recorded vs foreground       : {float(recorded)-explicit_fg:.6f}")
        print(f"Difference recorded vs all-class        : {float(recorded)-explicit_all:.6f}")

    # The exact all-class Dice is the most important test because a
    # background-inclusive metric can remain high even when all foreground
    # predictions collapse.
    if math.isfinite(float(recorded)):
        if abs(float(recorded) - explicit_fg) < 0.01:
            diagnosis = "RECORDED_METRIC_MATCHES_FOREGROUND_DICE"
        elif abs(float(recorded) - explicit_all) < 0.01:
            diagnosis = "RECORDED_METRIC_MATCHES_ALL_CLASS_DICE"
        elif abs(float(recorded) - bg) < 0.01:
            diagnosis = "RECORDED_METRIC_MATCHES_BACKGROUND_DICE"
        else:
            diagnosis = "RECORDED_METRIC_NOT_REPRODUCED_BY_EXPLICIT_DICE"
    else:
        diagnosis = "NO_RECORDED_CHECKPOINT_DICE"

    section("PART 24 DIAGNOSIS")
    print(f"Diagnosis                              : {diagnosis}")

    if diagnosis == "RECORDED_METRIC_MATCHES_FOREGROUND_DICE":
        print("âœ“ The checkpoint metric is consistent with explicit foreground Dice.")
    elif diagnosis == "RECORDED_METRIC_MATCHES_ALL_CLASS_DICE":
        print("CAUTION - recorded Dice appears to include background.")
        print("This can mask complete foreground collapse.")
    elif diagnosis == "RECORDED_METRIC_MATCHES_BACKGROUND_DICE":
        print("CRITICAL - recorded Dice appears dominated by background.")
    else:
        print("CRITICAL - Part 15 recorded Dice is not reproduced by the current")
        print("checkpoint using the explicit metric definitions tested.")

    section("SAVING PART 24 RESULTS")

    case_csv = OUTPUT_DIR / "part24_checkpoint_metric_integrity_case_metrics.csv"
    df.to_csv(case_csv, index=False)

    summary = {
        "phase": 4,
        "part": 24,
        "purpose": "training loss and checkpoint metric integrity audit",
        "evaluation_only": True,
        "training_performed": False,
        "weights_modified": False,
        "spider_used": False,
        "test_set_used": False,
        "project_root": str(PROJECT_ROOT),
        "rsna_root": str(RSNA_ROOT),
        "part11_source": str(PART11_SOURCE),
        "part15_source": str(part15_source) if part15_source else None,
        "part15_checkpoint": str(PART15_CHECKPOINT),
        "checkpoint_sha256": sha256_file(PART15_CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_dice": checkpoint.get("best_val_dice"),
        "checkpoint_seed": checkpoint.get("seed"),
        "checkpoint_num_classes": checkpoint.get("num_classes"),
        "checkpoint_feature_size": checkpoint.get("feature_size"),
        "checkpoint_patch_size": safe_json(checkpoint.get("patch_size")),
        "validation_cohort_size": len(val_cohort),
        "cases_audited": len(df),
        "case_errors": errors,
        "mean_foreground_dice": explicit_fg,
        "mean_all_class_dice": explicit_all,
        "mean_background_dice": bg,
        "empty_foreground_prediction_cases": int(
            (df["pred_foreground_voxels"] == 0).sum()
        ),
        "mean_pred_foreground_voxels": float(df["pred_foreground_voxels"].mean()),
        "mean_target_foreground_voxels": float(df["target_foreground_voxels"].mean()),
        "diagnosis": diagnosis,
        "source_audit": source_audit,
        "history_columns": list(history.columns),
        "history_records": history.to_dict(orient="records"),
    }

    summary_json = OUTPUT_DIR / "phase4_part24_training_loss_checkpoint_metric_integrity_summary.json"
    summary_json.write_text(
        json.dumps(safe_json(summary), indent=2),
        encoding="utf-8",
    )

    report = []
    report.append("PHASE 4 - PART 24")
    report.append("RSNA-ONLY TRAINING LOSS / CHECKPOINT METRIC INTEGRITY AUDIT")
    report.append("")
    report.append(f"Checkpoint: {PART15_CHECKPOINT}")
    report.append(f"Checkpoint SHA256: {sha256_file(PART15_CHECKPOINT)}")
    report.append(f"Cases audited: {len(df)}")
    report.append(f"Recorded checkpoint best_val_dice: {recorded}")
    report.append(f"Explicit foreground Dice: {explicit_fg:.6f}")
    report.append(f"Explicit all-class Dice: {explicit_all:.6f}")
    report.append(f"Mean background Dice: {bg:.6f}")
    report.append(
        f"Empty foreground predictions: "
        f"{int((df['pred_foreground_voxels'] == 0).sum())}/{len(df)}"
    )
    report.append(f"Diagnosis: {diagnosis}")
    report.append("")
    report.append("No training was performed. No weights were modified.")

    report_path = REPORT_DIR / "phase4_part24_training_loss_checkpoint_metric_integrity_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved: {case_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")

    section("PHASE 4 - PART 24 COMPLETE")
    print("No training performed.")
    print("No model weights modified.")
    print(f"Diagnosis: {diagnosis}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPART 24 interrupted by user.")
        raise
    except Exception as exc:
        section("PART 24 ERROR")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

