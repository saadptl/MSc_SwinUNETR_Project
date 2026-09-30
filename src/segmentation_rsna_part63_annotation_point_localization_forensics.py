"""
PHASE 4 - PART 63
RSNA ANNOTATION-POINT LOCALIZATION FORENSICS

Purpose
-------
Part 62 showed that low-threshold foreground responses did not provide
strong class-specific spatial localization when compared with the
R2 pseudo-mask regions.

Part 63 performs a more direct diagnostic:

    Does the predicted probability for the TRUE RSNA class become
    elevated at the ORIGINAL RSNA annotation point itself?

This avoids treating the entire sparse Gaussian pseudo-mask as if it
were a manual segmentation target.

Scientific scope
----------------
- RSNA validation cohort only.
- RSNA point coordinates from train_label_coordinates.csv.
- Existing Part 58 checkpoints only.
- No training.
- No optimizer.
- No backward pass.
- No model-weight modification.
- No SPIDER.
- No RSNA test set.
- Part 15 remains untouched.

For each annotation point, the script:
1. Loads the native image/mask through the validated Part 11 loader.
2. Applies the same final resize to (64,96,96).
3. Applies the same R2 foreground-centered crop used by Part 62.
4. Maps the ORIGINAL RSNA (x,y,instance) point into that crop.
5. Measures:
   - true-class probability at the point,
   - local maximum/mean probability in radii 1/2/3,
   - true-class rank among all six classes,
   - true-vs-background probability margin,
   - threshold hit rates,
   - random-control probability for comparison.
6. Aggregates by class and checkpoint.

Important limitation
--------------------
RSNA coordinates are point annotations, not manual segmentation masks.
A positive point-localization result would indicate localization signal,
not proof of correct medical segmentation.

Output
------
outputs/segmentation/rsna_part63_annotation_point_localization_forensics/
    reports/
        part63_annotation_point_metrics.csv
        part63_annotation_point_aggregate.csv
        part63_class_ranking.csv
        part63_checkpoint_summary.csv
        part63_summary.json
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

PART11 = ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9 = ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)
INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
VACSV = P15 / "part15_validation_cohort.csv"

P58 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part58_r2_lr_stability_confirmation"
)

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)
COORD_CSV = RSNA_ROOT / "train_label_coordinates.csv"

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part63_annotation_point_localization_forensics"
)
REPORT = OUT / "reports"


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================

N = 100
FULL = (64, 96, 96)
CROP = (32, 64, 64)
RADIUS = 2

# Thresholds deliberately include the Part 59/62 operating region.
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]

# Local spatial windows around the mapped RSNA point.
LOCAL_RADII = [0, 1, 2, 3]

EPOCHS = [1, 2, 3, 4, 5]

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASSES.items()}
CLASS_NAME_TO_ID.update({
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
})

CONDS = {
    "A_constant_5e5": P58 / "A_constant_5e5",
    "B_step_1e4_to_5e5": P58 / "B_step_1e4_to_5e5",
}


# ============================================================================
# HELPERS
# ============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 82)
    print(text)
    print("=" * 82)


def mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def ckpt(directory: Path, epoch: int) -> Path:
    candidates = (
        directory / f"epoch_{epoch:02d}.pth",
        directory / "checkpoints" / f"epoch_{epoch:02d}.pth",
    )
    for q in candidates:
        if q.exists():
            return q
    raise FileNotFoundError(
        f"Checkpoint not found for epoch {epoch}: {directory}"
    )


def load_model(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    payload = state
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(state.get(key), dict):
                payload = state[key]
                break

    payload = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in payload.items()
    }

    missing, unexpected = model.load_state_dict(
        payload,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

    model.eval()


def resize_3d(p11: Any, array: np.ndarray, is_mask: bool) -> np.ndarray:
    """
    Use the exact Part 11 resize implementation.
    """
    return p11.resize_3d(
        array,
        FULL,
        is_mask=is_mask,
    )


def dil6(a: np.ndarray, n: int) -> np.ndarray:
    a = a.astype(bool).copy()
    for _ in range(n):
        b = a.copy()
        b[1:] |= a[:-1]
        b[:-1] |= a[1:]
        b[:, 1:] |= a[:, :-1]
        b[:, :-1] |= a[:, 1:]
        b[:, :, 1:] |= a[:, :, :-1]
        b[:, :, :-1] |= a[:, :, 1:]
        a = b
    return a


def dilate_labels(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Reproduce the Part 62 R2 label dilation / deterministic overlap handling.
    """
    if n == 0:
        return mask.astype(np.int64).copy()

    nc = 6
    parts = []

    for c in range(1, nc):
        q = mask == c
        if q.any():
            parts.append((int(q.sum()), c, dil6(q, n)))

    parts.sort(key=lambda x: (-x[0], x[1]))

    out = np.zeros_like(mask, dtype=np.int64)
    occupied = np.zeros_like(mask, dtype=bool)

    for _, c, q in parts:
        q = q & ~occupied
        out[q] = c
        occupied |= q

    return out


def centered_crop(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    q = np.argwhere(mask > 0)

    if q.size:
        center = np.round(q.mean(axis=0)).astype(int)
    else:
        center = np.array(
            [FULL[i] // 2 for i in range(3)],
            dtype=int,
        )

    start = [
        max(
            0,
            min(
                int(center[i]) - CROP[i] // 2,
                FULL[i] - CROP[i],
            ),
        )
        for i in range(3)
    ]

    z, y, x = start
    dz, dy, dx = CROP

    return (
        image[z:z + dz, y:y + dy, x:x + dx].astype(np.float32),
        mask[z:z + dz, y:y + dy, x:x + dx].astype(np.int64),
        (z, y, x),
    )


def resize_coordinate(
    x: float,
    y: float,
    z: float,
    native_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """
    Approximate the coordinate mapping used by torch.interpolate
    with align_corners=False.

    Coordinates are represented as voxel centers.
    """
    nz, nh, nw = [float(v) for v in native_shape]
    tz, th, tw = [float(v) for v in target_shape]

    xo = (float(x) + 0.5) * tw / nw - 0.5
    yo = (float(y) + 0.5) * th / nh - 0.5
    zo = (float(z) + 0.5) * tz / nz - 0.5

    return zo, yo, xo


def local_values(
    volume: np.ndarray,
    point: Tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    z, y, x = point
    dz0 = max(0, z - radius)
    dz1 = min(volume.shape[0], z + radius + 1)
    dy0 = max(0, y - radius)
    dy1 = min(volume.shape[1], y + radius + 1)
    dx0 = max(0, x - radius)
    dx1 = min(volume.shape[2], x + radius + 1)

    return volume[dz0:dz1, dy0:dy1, dx0:dx1]


def map_annotation_points(
    p11: Any,
    p9: Any,
    val_df: pd.DataFrame,
    coord_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Load each validation series once and map all original RSNA points into
    the exact Part 62 evaluation crop.

    Native -> FULL resize is performed by Part 11.
    Crop origin is determined from the R2 pseudo-mask exactly as in Part 62.
    """
    mapped = []

    for case_no, (_, row) in enumerate(val_df.iterrows(), 1):
        image_native, mask_native, info = p11.load_case_robust(
            row,
            p9,
        )

        image_native = np.asarray(image_native, dtype=np.float32)
        mask_native = np.asarray(mask_native, dtype=np.int64)

        native_shape = tuple(int(v) for v in image_native.shape)

        image_full = resize_3d(
            p11,
            image_native,
            is_mask=False,
        )
        mask_full = resize_3d(
            p11,
            mask_native,
            is_mask=True,
        )

        mask_eval = dilate_labels(mask_full, RADIUS)

        _, _, crop_start = centered_crop(
            image_full,
            mask_eval,
        )

        study_id = str(row["study_id"])
        series_id = str(row["series_id"])

        ann = coord_df[
            (coord_df["study_id"].astype(str) == study_id)
            & (coord_df["series_id"].astype(str) == series_id)
        ].copy()

        if ann.empty:
            print(
                f"WARNING: no RSNA coordinate rows for validation "
                f"case {case_no}: {study_id}/{series_id}"
            )
            continue

        # Part 11 DICOM order is by InstanceNumber. Build an instance -> z map.
        dicom_records = p11.read_dicom_series_robust(
            p11.resolve_series_dir(row)
        )[1]

        instance_to_z = {
            int(ds.get("InstanceNumber", 0)): int(z)
            for z, ds in enumerate(dicom_records)
        }

        for _, a in ann.iterrows():
            try:
                x = float(a["x"])
                y = float(a["y"])
                instance = int(float(a["instance_number"]))
            except Exception:
                continue

            condition = str(a["condition"]).strip()
            cid = CLASS_NAME_TO_ID.get(condition)

            if cid is None:
                continue

            if instance not in instance_to_z:
                continue

            native_z = instance_to_z[instance]

            if not (
                0 <= x < native_shape[2]
                and 0 <= y < native_shape[1]
            ):
                continue

            zf, yf, xf = resize_coordinate(
                x=x,
                y=y,
                z=native_z,
                native_shape=native_shape,
                target_shape=FULL,
            )

            cz, cy, cx = crop_start

            zc = zf - cz
            yc = yf - cy
            xc = xf - cx

            # The annotation can lie just outside the final crop.
            # Keep it only when its nearest voxel is inside the crop.
            zi = int(round(zc))
            yi = int(round(yc))
            xi = int(round(xc))

            if not (
                0 <= zi < CROP[0]
                and 0 <= yi < CROP[1]
                and 0 <= xi < CROP[2]
            ):
                continue

            mapped.append({
                "case_index": case_no,
                "study_id": study_id,
                "series_id": series_id,
                "condition": condition,
                "class_id": cid,
                "level": str(a.get("level", "")),
                "instance_number": instance,
                "native_x": x,
                "native_y": y,
                "native_z": native_z,
                "mapped_z_float": float(zc),
                "mapped_y_float": float(yc),
                "mapped_x_float": float(xc),
                "z": zi,
                "y": yi,
                "x": xi,
                "native_shape": str(native_shape),
                "crop_start": str(crop_start),
                "point_inside_crop": True,
            })

        if case_no in (1, 25, 50, 75, 100):
            print(
                f"VALIDATION {case_no:03d}/{len(val_df)} "
                f"annotations={len(ann)} "
                f"native={native_shape} "
                f"crop_start={crop_start}"
            )

    return mapped


def extract_probability_maps(
    model: torch.nn.Module,
    cases: List[Dict[str, Any]],
    device: torch.device,
) -> List[np.ndarray]:
    out = []

    for case in cases:
        x = (
            torch.from_numpy(case["image"])
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        out.append(probs)
        del x, logits

    return out


def build_cases(
    p11: Any,
    p9: Any,
    val_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    cases = []

    for case_no, (_, row) in enumerate(val_df.iterrows(), 1):
        image_native, mask_native, _ = p11.load_case_robust(
            row,
            p9,
        )

        image_native = np.asarray(image_native, dtype=np.float32)
        mask_native = np.asarray(mask_native, dtype=np.int64)

        image_full = resize_3d(
            p11,
            image_native,
            is_mask=False,
        )
        mask_full = resize_3d(
            p11,
            mask_native,
            is_mask=True,
        )

        mask_eval = dilate_labels(mask_full, RADIUS)

        image_crop, mask_crop, crop_start = centered_crop(
            image_full,
            mask_eval,
        )

        cases.append({
            "index": case_no,
            "image": image_crop,
            "mask": mask_crop,
            "crop_start": crop_start,
        })

    return cases


def load_coordinate_csv() -> pd.DataFrame:
    df = pd.read_csv(COORD_CSV)

    required = {
        "study_id",
        "series_id",
        "condition",
        "level",
        "instance_number",
        "x",
        "y",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Coordinate CSV missing columns: {sorted(missing)}"
        )

    for c in ("study_id", "series_id", "condition", "level"):
        df[c] = df[c].astype(str)

    for c in ("instance_number", "x", "y"):
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "study_id",
            "series_id",
            "condition",
            "instance_number",
            "x",
            "y",
        ]
    ).copy()

    return df


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    banner("PART 63 PATH VALIDATION")

    required_paths = [
        ("Project root", ROOT),
        ("Part 11", PART11),
        ("Part 9", PART9),
        ("Part 15 initialization", INIT),
        ("Part 15 validation cohort", VACSV),
        ("RSNA coordinate CSV", COORD_CSV),
        ("Part 58 output", P58),
    ]

    for name, path in required_paths:
        print(
            f"{name:<40}: "
            f"{'FOUND' if path.exists() else 'MISSING'}"
        )
        if not path.exists():
            raise FileNotFoundError(path)

    for condition, directory in CONDS.items():
        print(
            f"{condition + ' directory':<40}: "
            f"{'FOUND' if directory.exists() else 'MISSING'}"
        )
        if not directory.exists():
            raise FileNotFoundError(directory)

        for epoch in EPOCHS:
            ckpt(directory, epoch)

    banner("PART 63 — RSNA ANNOTATION-POINT LOCALIZATION FORENSICS")

    print(
        f"Validation cohort : {N}\n"
        f"Pseudo-mask radius : R{RADIUS}\n"
        f"Full volume : {FULL}\n"
        f"Crop : {CROP}\n"
        f"Thresholds : {THRESHOLDS}\n"
        f"Local radii : {LOCAL_RADII}\n"
        f"Checkpoints : epochs {EPOCHS}\n"
        f"Point coordinates : RSNA train_label_coordinates.csv\n"
        f"Training performed : NO\n"
        f"Optimizer used : NO\n"
        f"Backward pass : NO\n"
        f"SPIDER : NO\n"
        f"Test set : NO\n"
        f"Part 15 modified : NO\n"
        f"Initialization SHA256 : {sha(INIT)}"
    )

    p11 = mod(PART11, "p11_part63")
    p9 = mod(PART9, "p9_part63")

    val_df = pd.read_csv(VACSV).head(N)
    coord_df = load_coordinate_csv()

    banner("BUILDING VALIDATION CASES")
    cases = build_cases(
        p11,
        p9,
        val_df,
    )

    if not cases:
        raise RuntimeError("No validation cases loaded.")

    print(
        f"Validation tensors loaded : {len(cases)}"
    )
    print(
        f"First image shape : {cases[0]['image'].shape}"
    )
    print(
        f"First mask shape  : {cases[0]['mask'].shape}"
    )
    print(
        f"First labels      : "
        f"{np.unique(cases[0]['mask']).tolist()}"
    )

    assert tuple(cases[0]["image"].shape) == CROP
    assert tuple(cases[0]["mask"].shape) == CROP

    banner("PART 63 SHAPE / LABEL SMOKE TEST")
    print("✓ Shape / label smoke test PASSED.")

    banner("MAPPING ORIGINAL RSNA ANNOTATION POINTS")

    point_rows = map_annotation_points(
        p11,
        p9,
        val_df,
        coord_df,
    )

    if not point_rows:
        raise RuntimeError(
            "No annotation points could be mapped into the evaluation crop."
        )

    point_df = pd.DataFrame(point_rows)

    print(
        f"Mapped annotation points : {len(point_df)}"
    )

    print(
        "Mapped points by class:"
    )
    for cid, cname in CLASSES.items():
        count = int(
            (point_df["class_id"] == cid).sum()
        )
        print(
            f"  C{cid} {cname:<36}: {count}"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"\nPyTorch : {torch.__version__}"
        f"\nDevice : {device}"
    )

    all_rows: List[Dict[str, Any]] = []

    # Cache class-independent random control offsets.
    rng = np.random.default_rng(6301)

    for condition, directory in CONDS.items():
        banner(f"PART 63 CONDITION: {condition}")

        model = p11.create_model(device)
        model.to(device)

        for epoch in EPOCHS:
            checkpoint = ckpt(
                directory,
                epoch,
            )

            load_model(
                model,
                checkpoint,
                device,
            )

            print(
                f"\nCHECKPOINT EPOCH {epoch:02d} "
                f"| SHA256={sha(checkpoint)}"
            )

            probs = extract_probability_maps(
                model,
                cases,
                device,
            )

            # Map case index -> probability tensor.
            for _, point in point_df.iterrows():
                case_idx = int(point["case_index"]) - 1
                cid = int(point["class_id"])
                z = int(point["z"])
                y = int(point["y"])
                x = int(point["x"])

                pmap = probs[case_idx]

                class_at_point = float(
                    pmap[cid, z, y, x]
                )
                bg_at_point = float(
                    pmap[0, z, y, x]
                )

                ranking = np.argsort(
                    pmap[:, z, y, x]
                )[::-1]

                rank_position = int(
                    np.where(ranking == cid)[0][0]
                ) + 1

                row_out = {
                    "condition": condition,
                    "epoch": epoch,
                    "checkpoint_sha256": sha(checkpoint),
                    "case_index": int(point["case_index"]),
                    "study_id": point["study_id"],
                    "series_id": point["series_id"],
                    "class_id": cid,
                    "class_name": CLASSES[cid],
                    "level": point["level"],
                    "instance_number": int(point["instance_number"]),
                    "z": z,
                    "y": y,
                    "x": x,
                    "true_class_probability": class_at_point,
                    "background_probability": bg_at_point,
                    "true_vs_background_margin": (
                        class_at_point - bg_at_point
                    ),
                    "true_class_rank": rank_position,
                    "top_class_id": int(ranking[0]),
                    "top_class_probability": float(
                        pmap[ranking[0], z, y, x]
                    ),
                }

                for radius in LOCAL_RADII:
                    vals = local_values(
                        pmap[cid],
                        (z, y, x),
                        radius,
                    )

                    row_out[
                        f"class_prob_local_mean_r{radius}"
                    ] = float(np.mean(vals))

                    row_out[
                        f"class_prob_local_max_r{radius}"
                    ] = float(np.max(vals))

                    row_out[
                        f"class_prob_local_median_r{radius}"
                    ] = float(np.median(vals))

                # Threshold hits at the point and in local max windows.
                for threshold in THRESHOLDS:
                    key = f"{threshold:.2f}"

                    row_out[
                        f"point_hit_t{key}"
                    ] = bool(
                        class_at_point >= threshold
                    )

                    local_r2 = local_values(
                        pmap[cid],
                        (z, y, x),
                        2,
                    )

                    row_out[
                        f"local_r2_hit_t{key}"
                    ] = bool(
                        float(np.max(local_r2))
                        >= threshold
                    )

                # Random control: same class, random voxel in same case.
                # This estimates whether the annotation point has higher
                # class probability than a generic location.
                rz = int(rng.integers(0, CROP[0]))
                ry = int(rng.integers(0, CROP[1]))
                rx = int(rng.integers(0, CROP[2]))

                random_prob = float(
                    pmap[cid, rz, ry, rx]
                )

                row_out[
                    "random_control_class_probability"
                ] = random_prob

                row_out[
                    "point_minus_random_probability"
                ] = (
                    class_at_point
                    - random_prob
                )

                all_rows.append(row_out)

            del probs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model
        gc.collect()

    metrics_df = pd.DataFrame(all_rows)

    banner("PART 63 CHECKPOINT SUMMARY")

    aggregate_rows: List[Dict[str, Any]] = []

    for condition in CONDS:
        for epoch in EPOCHS:
            sub = metrics_df[
                (metrics_df["condition"] == condition)
                & (metrics_df["epoch"] == epoch)
            ]

            if sub.empty:
                continue

            for cid, cname in CLASSES.items():
                q = sub[sub["class_id"] == cid]

                if q.empty:
                    continue

                aggregate_rows.append({
                    "condition": condition,
                    "epoch": epoch,
                    "class_id": cid,
                    "class_name": cname,
                    "annotation_points": len(q),
                    "mean_point_probability": float(
                        q["true_class_probability"].mean()
                    ),
                    "median_point_probability": float(
                        q["true_class_probability"].median()
                    ),
                    "mean_local_r2_max_probability": float(
                        q["class_prob_local_max_r2"].mean()
                    ),
                    "mean_local_r3_max_probability": float(
                        q["class_prob_local_max_r3"].mean()
                    ),
                    "mean_true_vs_background_margin": float(
                        q["true_vs_background_margin"].mean()
                    ),
                    "mean_point_minus_random_probability": float(
                        q["point_minus_random_probability"].mean()
                    ),
                    "mean_true_class_rank": float(
                        q["true_class_rank"].mean()
                    ),
                    "top1_accuracy": float(
                        (q["top_class_id"] == cid).mean()
                    ),
                    "point_hit_t010": float(
                        q["point_hit_t0.10"].mean()
                    ),
                    "point_hit_t015": float(
                        q["point_hit_t0.15"].mean()
                    ),
                    "point_hit_t020": float(
                        q["point_hit_t0.20"].mean()
                    ),
                    "point_hit_t025": float(
                        q["point_hit_t0.25"].mean()
                    ),
                    "point_hit_t030": float(
                        q["point_hit_t0.30"].mean()
                    ),
                    "local_r2_hit_t020": float(
                        q["local_r2_hit_t0.20"].mean()
                    ),
                })

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    for _, r in aggregate_df.iterrows():
        print(
            f"{r['condition']:<24} "
            f"E{int(r['epoch'])} "
            f"C{int(r['class_id'])} "
            f"{r['class_name']:<34} "
            f"PointP={r['mean_point_probability']:.4f} "
            f"R2Max={r['mean_local_r2_max_probability']:.4f} "
            f"Margin={r['mean_true_vs_background_margin']:.4f} "
            f"Rank={r['mean_true_class_rank']:.2f} "
            f"Top1={r['top1_accuracy']:.3f} "
            f"Hit@.20={r['point_hit_t020']:.3f}"
        )

    banner("PART 63 CLASS RANKING")

    ranking_rows: List[Dict[str, Any]] = []

    for cid, cname in CLASSES.items():
        q = aggregate_df[
            aggregate_df["class_id"] == cid
        ]

        if q.empty:
            continue

        # For each condition take the strongest epoch by point probability.
        condition_best = []

        for condition in CONDS:
            z = q[
                q["condition"] == condition
            ]

            if z.empty:
                continue

            best = z.loc[
                z["mean_point_probability"].idxmax()
            ]

            condition_best.append(best)

        if not condition_best:
            continue

        ranking_rows.append({
            "class_id": cid,
            "class_name": cname,
            "mean_best_point_probability": float(
                np.mean([
                    float(x["mean_point_probability"])
                    for x in condition_best
                ])
            ),
            "mean_best_local_r2_max_probability": float(
                np.mean([
                    float(x["mean_local_r2_max_probability"])
                    for x in condition_best
                ])
            ),
            "mean_best_point_minus_random": float(
                np.mean([
                    float(x["mean_point_minus_random_probability"])
                    for x in condition_best
                ])
            ),
            "mean_best_top1_accuracy": float(
                np.mean([
                    float(x["top1_accuracy"])
                    for x in condition_best
                ])
            ),
            "mean_best_hit_at_020": float(
                np.mean([
                    float(x["point_hit_t020"])
                    for x in condition_best
                ])
            ),
        })

    ranking_rows.sort(
        key=lambda x: (
            x["mean_best_point_probability"],
            x["mean_best_point_minus_random"],
        ),
        reverse=True,
    )

    for rank, r in enumerate(ranking_rows, 1):
        print(
            f"{rank}. C{r['class_id']} "
            f"{r['class_name']:<34} "
            f"PointP={r['mean_best_point_probability']:.4f} "
            f"R2Max={r['mean_best_local_r2_max_probability']:.4f} "
            f"Point-Random={r['mean_best_point_minus_random']:.4f} "
            f"Top1={r['mean_best_top1_accuracy']:.3f}"
        )

    banner("PART 63 DIAGNOSTIC INTERPRETATION")

    mean_point = float(
        metrics_df["true_class_probability"].mean()
    )

    mean_random = float(
        metrics_df["random_control_class_probability"].mean()
    )

    mean_delta = float(
        metrics_df["point_minus_random_probability"].mean()
    )

    top1 = float(
        (
            metrics_df["top_class_id"]
            == metrics_df["class_id"]
        ).mean()
    )

    hit20 = float(
        metrics_df["point_hit_t0.20"].mean()
    )

    # Conservative diagnostic thresholds.
    if (
        mean_delta >= 0.08
        and top1 >= 0.40
        and hit20 >= 0.40
    ):
        diagnosis = "DIRECT_RSNA_POINT_LOCALIZATION_SIGNAL_IS_EVIDENT"
    elif (
        mean_delta >= 0.03
        or top1 >= 0.25
        or hit20 >= 0.25
    ):
        diagnosis = "WEAK_DIRECT_POINT_LOCALIZATION_SIGNAL"
    else:
        diagnosis = "NO_STRONG_DIRECT_POINT_LOCALIZATION_SIGNAL"

    print(
        f"Mean true-class point probability : {mean_point:.6f}\n"
        f"Mean random-control probability  : {mean_random:.6f}\n"
        f"Mean point-minus-random          : {mean_delta:.6f}\n"
        f"True-class Top-1 rate             : {top1:.6f}\n"
        f"Point hit rate @ T=0.20           : {hit20:.6f}\n"
        f"Diagnosis                         : {diagnosis}"
    )

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_csv = REPORT / "part63_annotation_point_metrics.csv"
    agg_csv = REPORT / "part63_annotation_point_aggregate.csv"
    rank_csv = REPORT / "part63_class_ranking.csv"
    ckpt_csv = REPORT / "part63_checkpoint_summary.csv"
    summary_json = REPORT / "part63_summary.json"

    metrics_df.to_csv(
        point_csv,
        index=False,
    )

    aggregate_df.to_csv(
        agg_csv,
        index=False,
    )

    pd.DataFrame(
        ranking_rows
    ).to_csv(
        rank_csv,
        index=False,
    )

    checkpoint_rows = []

    for (condition, epoch), q in metrics_df.groupby(
        ["condition", "epoch"]
    ):
        checkpoint_rows.append({
            "condition": condition,
            "epoch": int(epoch),
            "annotation_points": len(q),
            "mean_point_probability": float(
                q["true_class_probability"].mean()
            ),
            "mean_local_r2_max_probability": float(
                q["class_prob_local_max_r2"].mean()
            ),
            "mean_point_minus_random_probability": float(
                q["point_minus_random_probability"].mean()
            ),
            "mean_true_vs_background_margin": float(
                q["true_vs_background_margin"].mean()
            ),
            "true_class_top1_accuracy": float(
                (
                    q["top_class_id"]
                    == q["class_id"]
                ).mean()
            ),
            "point_hit_t020": float(
                q["point_hit_t0.20"].mean()
            ),
            "local_r2_hit_t020": float(
                q["local_r2_hit_t0.20"].mean()
            ),
        })

    checkpoint_df = pd.DataFrame(
        checkpoint_rows
    )

    checkpoint_df.to_csv(
        ckpt_csv,
        index=False,
    )

    summary = {
        "part": 63,
        "purpose": (
            "Determine whether low-threshold model responses are "
            "elevated directly at original RSNA annotation points."
        ),
        "validation_cases": N,
        "mapped_annotation_points": int(len(metrics_df)),
        "pseudo_mask_radius": RADIUS,
        "full_volume": FULL,
        "crop": CROP,
        "thresholds": THRESHOLDS,
        "local_radii": LOCAL_RADII,
        "epochs": EPOCHS,
        "conditions": list(CONDS.keys()),
        "training_performed": False,
        "optimizer_used": False,
        "backward_pass": False,
        "spider": False,
        "test_set": False,
        "part15_modified": False,
        "initialization_sha256": sha(INIT),
        "mean_true_class_point_probability": mean_point,
        "mean_random_control_probability": mean_random,
        "mean_point_minus_random_probability": mean_delta,
        "true_class_top1_accuracy": top1,
        "point_hit_rate_t020": hit20,
        "diagnosis": diagnosis,
        "scientific_note": (
            "RSNA annotations are point annotations, not manual "
            "pixel-wise segmentation masks. Direct point localization "
            "therefore indicates localization signal only and does "
            "not establish medical ground-truth segmentation accuracy."
        ),
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    banner("PART 63 COMPLETE")

    print(
        f"Output directory : {OUT}\n"
        f"Point metrics CSV : {point_csv}\n"
        f"Aggregate CSV : {agg_csv}\n"
        f"Class ranking CSV : {rank_csv}\n"
        f"Checkpoint summary CSV : {ckpt_csv}\n"
        f"Summary JSON : {summary_json}"
    )


if __name__ == "__main__":
    main()
