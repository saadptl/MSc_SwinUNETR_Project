"""
PART 75
RSNA-ONLY C2/C3 SUPERVISION + ANATOMICAL LOCALIZATION AUDIT

Purpose
-------
Part 74 established that C2/C3 probability activation is spatially
misplaced or very weak and that C2/C3 mutual confusion is not the
dominant failure mode.

Part 75 therefore performs an evaluation-only audit of the supervision
and anatomical localization relationship for C2/C3.

It compares, per validation case:
    1. RSNA annotation point locations
    2. R2 pseudo-mask locations
    3. C2/C3 target pseudo-mask geometry
    4. C2/C3 model probability geometry
    5. distance from annotation points to target pseudo-mask
    6. distance from annotation points to high-probability predictions
    7. distance from pseudo-mask to high-probability predictions
    8. crop inclusion / exclusion
    9. prediction volume and bounding-box geometry

IMPORTANT
---------
This script is forensic only:
- No training
- No optimizer
- No checkpoint modification
- No SPIDER
- No RSNA test set
- First 50 Part 15 validation cases
- R2_FULL
- Part 71 class-balanced E3 checkpoint

The RSNA annotations are treated as point/localizer annotations.
They are NOT treated as segmentation ground truth.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


# ============================================================================
# PATHS
# ============================================================================
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent

P11 = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)
VAL_CSV = P15 / "part15_validation_cohort.csv"

P71 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part71_class_balanced_dicece_training"
)
CKPT = (
    P71
    / "checkpoints"
    / "part71_r2_full_class_balanced_epoch3.pth"
)

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part75_c2_c3_supervision_localization_audit"
)
REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================
VAL_N = 50

FULL = (64, 96, 96)
CROP = (32, 64, 64)

NUM_CLASSES = 6

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

FOCUS_CLASSES = [2, 3]

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

# Part 73 / 74 indicated these are the most informative thresholds.
BEST_THRESHOLDS = {
    2: 0.15,
    3: 0.20,
}

# Also audit the top probability voxels because threshold masks are
# extremely oversized for C2/C3.
TOP_FRACTIONS = [
    0.001,
    0.005,
    0.01,
]

STRUCTURE_6 = ndimage.generate_binary_structure(
    3,
    1,
)


# ============================================================================
# UTILITIES
# ============================================================================
def banner(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def reset_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def safe_div(
    a: float,
    b: float,
) -> float:
    return (
        float(a / b)
        if b > 0
        else 0.0
    )


def percentile_or_zero(
    x: np.ndarray,
    q: float,
) -> float:
    if x.size == 0:
        return 0.0

    return float(
        np.percentile(
            x,
            q,
        )
    )


# ============================================================================
# R2 PSEUDO-MASK
# ============================================================================
def dilate_labels(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    mask = np.asarray(
        mask,
        dtype=np.int64,
    )

    if radius == 0:
        return mask.copy()

    result = mask.copy()

    for c in range(
        1,
        NUM_CLASSES,
    ):
        binary = mask == c

        if not binary.any():
            continue

        structure = np.ones(
            (2 * radius + 1,) * 3,
            dtype=bool,
        )

        expanded = ndimage.binary_dilation(
            binary,
            structure=structure,
        )

        # Preserve already assigned labels.
        result[
            expanded
            & (result == 0)
        ] = c

    return result


def centered_crop(
    mask: np.ndarray,
    center: np.ndarray,
    shape: tuple[int, int, int],
):
    starts = []

    for dim, size, c in zip(
        mask.shape,
        shape,
        center,
    ):
        start = int(
            round(
                float(c)
                - size / 2.0
            )
        )

        start = max(
            0,
            min(
                start,
                dim - size,
            ),
        )

        starts.append(start)

    z0, y0, x0 = starts

    crop = mask[
        z0:z0 + shape[0],
        y0:y0 + shape[1],
        x0:x0 + shape[2],
    ]

    return crop, tuple(starts)


# ============================================================================
# RSNA ANNOTATION EXTRACTION
# ============================================================================
def first_value(
    row: pd.Series,
    candidates: list[str],
):
    for key in candidates:
        if key in row.index:
            value = row[key]

            if (
                pd.notna(value)
                and str(value).strip() != ""
            ):
                return value

    return None


def as_id(
    value: Any,
):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def resolve_series_dir(
    row: pd.Series,
) -> Path | None:
    candidates = [
        "series_dir",
        "series_path",
        "dicom_dir",
        "image_dir",
        "path",
    ]

    value = first_value(
        row,
        candidates,
    )

    if value is None:
        return None

    p = Path(
        str(value)
    )

    if p.exists():
        return p

    # Some cohort CSVs contain paths relative to project root.
    p2 = ROOT / p

    if p2.exists():
        return p2

    return None


def find_numeric_column(
    columns: list[str],
    tokens: list[str],
):
    lower = {
        c.lower(): c
        for c in columns
    }

    for c_lower, original in lower.items():
        if all(
            token in c_lower
            for token in tokens
        ):
            return original

    return None


def extract_annotation_table(
    row: pd.Series,
) -> pd.DataFrame:
    """
    Extract RSNA point/localizer annotations from common Part 15 cohort
    representations.

    Supported forms:
    - an annotation CSV/path column
    - serialized annotation dictionaries/lists
    - direct x/y/z + class/level columns

    If no annotation representation is available in the cohort row,
    return an empty table rather than inventing coordinates.
    """

    # ------------------------------------------------------------------------
    # Direct annotation object/path columns
    # ------------------------------------------------------------------------
    annotation_value = first_value(
        row,
        [
            "annotations",
            "annotation",
            "annotation_path",
            "annotation_file",
            "annotations_path",
            "point_annotations",
            "points",
        ],
    )

    if annotation_value is not None:
        # Path to an annotation CSV.
        p = Path(
            str(annotation_value)
        )

        candidates = [
            p,
            ROOT / p,
        ]

        for candidate in candidates:
            if (
                candidate.exists()
                and candidate.is_file()
                and candidate.suffix.lower()
                == ".csv"
            ):
                try:
                    ann = pd.read_csv(
                        candidate
                    )

                    return normalize_annotation_dataframe(
                        ann
                    )
                except Exception:
                    pass

        # Serialized JSON-like representation.
        if isinstance(
            annotation_value,
            (list, tuple, dict),
        ):
            try:
                ann = pd.DataFrame(
                    annotation_value
                    if isinstance(
                        annotation_value,
                        (list, tuple),
                    )
                    else [annotation_value]
                )

                return normalize_annotation_dataframe(
                    ann
                )
            except Exception:
                pass

        text = str(
            annotation_value
        ).strip()

        if text.startswith(
            "["
        ) or text.startswith(
            "{"
        ):
            try:
                parsed = json.loads(
                    text
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    parsed = [parsed]

                ann = pd.DataFrame(
                    parsed
                )

                return normalize_annotation_dataframe(
                    ann
                )
            except Exception:
                pass

    # ------------------------------------------------------------------------
    # Direct x/y/z + class columns in cohort.
    # ------------------------------------------------------------------------
    cols = list(
        row.index
    )

    x_col = (
        find_numeric_column(
            cols,
            ["x"],
        )
        or find_numeric_column(
            cols,
            ["coord", "x"],
        )
    )

    y_col = (
        find_numeric_column(
            cols,
            ["y"],
        )
        or find_numeric_column(
            cols,
            ["coord", "y"],
        )
    )

    z_col = (
        find_numeric_column(
            cols,
            ["z"],
        )
        or find_numeric_column(
            cols,
            ["coord", "z"],
        )
    )

    class_col = None

    for token in [
        "class",
        "condition",
        "label",
        "finding",
        "diagnosis",
    ]:
        class_col = find_numeric_column(
            cols,
            [token],
        )

        if class_col is not None:
            break

    if (
        x_col is not None
        and y_col is not None
        and z_col is not None
    ):
        values = {
            "x": row[x_col],
            "y": row[y_col],
            "z": row[z_col],
        }

        if class_col is not None:
            values["class"] = row[
                class_col
            ]

        ann = pd.DataFrame(
            [values]
        )

        return normalize_annotation_dataframe(
            ann
        )

    return pd.DataFrame(
        columns=[
            "x",
            "y",
            "z",
            "class",
        ]
    )


def normalize_annotation_dataframe(
    ann: pd.DataFrame,
) -> pd.DataFrame:
    if ann is None or ann.empty:
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "class",
            ]
        )

    columns = list(
        ann.columns
    )

    def locate(
        options: list[str],
    ):
        for option in options:
            for col in columns:
                if str(col).lower() == option:
                    return col

        for option in options:
            for col in columns:
                if option in str(
                    col
                ).lower():
                    return col

        return None

    x_col = locate(
        [
            "x",
            "x_coord",
            "xcoordinate",
            "coordinate_x",
        ]
    )

    y_col = locate(
        [
            "y",
            "y_coord",
            "ycoordinate",
            "coordinate_y",
        ]
    )

    z_col = locate(
        [
            "z",
            "z_coord",
            "zcoordinate",
            "coordinate_z",
        ]
    )

    class_col = locate(
        [
            "class",
            "class_id",
            "label",
            "condition",
            "finding",
        ]
    )

    if (
        x_col is None
        or y_col is None
        or z_col is None
    ):
        return pd.DataFrame(
            columns=[
                "x",
                "y",
                "z",
                "class",
            ]
        )

    result = pd.DataFrame()

    result["x"] = pd.to_numeric(
        ann[x_col],
        errors="coerce",
    )

    result["y"] = pd.to_numeric(
        ann[y_col],
        errors="coerce",
    )

    result["z"] = pd.to_numeric(
        ann[z_col],
        errors="coerce",
    )

    if class_col is not None:
        result["class"] = ann[
            class_col
        ]
    else:
        result["class"] = ""

    result = result.dropna(
        subset=[
            "x",
            "y",
            "z",
        ]
    )

    return result.reset_index(
        drop=True
    )


def find_annotation_file(
    row: pd.Series,
) -> Path | None:
    """
    Search only locations associated with the current case.
    No broad filesystem search is performed.
    """

    series_dir = resolve_series_dir(
        row
    )

    if series_dir is None:
        return None

    candidates = [
        series_dir / "annotations.csv",
        series_dir / "annotation.csv",
        series_dir / "points.csv",
        series_dir.parent / "annotations.csv",
        series_dir.parent / "annotation.csv",
        series_dir.parent / "points.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


# ============================================================================
# LABEL / CLASS MAPPING
# ============================================================================
def map_annotation_class(
    value: Any,
) -> int | None:
    if value is None:
        return None

    text = str(
        value
    ).strip().lower()

    if not text:
        return None

    # Numeric class IDs.
    try:
        numeric = int(
            float(text)
        )

        if numeric in FOCUS_CLASSES:
            return numeric

    except Exception:
        pass

    # Common RSNA condition naming.
    if (
        "left"
        in text
        and (
            "foraminal"
            in text
            or "neural foraminal"
            in text
        )
    ):
        return 2

    if (
        "right"
        in text
        and (
            "foraminal"
            in text
            or "neural foraminal"
            in text
        )
    ):
        return 3

    return None


# ============================================================================
# GEOMETRY
# ============================================================================
def bbox(
    binary: np.ndarray,
):
    q = np.argwhere(
        binary
    )

    if len(q) == 0:
        return None

    lo = q.min(
        axis=0
    )

    hi = q.max(
        axis=0
    )

    return (
        tuple(
            int(x)
            for x in lo
        ),
        tuple(
            int(x)
            for x in hi
        ),
    )


def centroid(
    binary: np.ndarray,
):
    q = np.argwhere(
        binary
    )

    if len(q) == 0:
        return None

    return tuple(
        float(x)
        for x in q.mean(
            axis=0
        )
    )


def point_inside(
    point_zyx: np.ndarray,
    shape: tuple[int, int, int],
) -> bool:
    return bool(
        np.all(
            point_zyx >= 0
        )
        and np.all(
            point_zyx
            < np.asarray(
                shape
            )
        )
    )


def nearest_distance_to_mask(
    point: np.ndarray,
    binary: np.ndarray,
) -> float:
    q = np.argwhere(
        binary
    )

    if len(q) == 0:
        return -1.0

    return float(
        np.linalg.norm(
            q
            - point[None, :],
            axis=1,
        ).min()
    )


def mean_nearest_point_distance(
    points: np.ndarray,
    binary: np.ndarray,
) -> float:
    if (
        points.size == 0
        or not binary.any()
    ):
        return -1.0

    q = np.argwhere(
        binary
    )

    distances = []

    for p in points:
        distances.append(
            np.linalg.norm(
                q - p[None, :],
                axis=1,
            ).min()
        )

    return float(
        np.mean(
            distances
        )
    )


def probability_weighted_centroid(
    probability: np.ndarray,
):
    total = float(
        probability.sum()
    )

    if total <= 0:
        return None

    coords = np.indices(
        probability.shape
    )

    return tuple(
        float(
            (coords[d] * probability).sum()
            / total
        )
        for d in range(3)
    )


def top_probability_mask(
    probability: np.ndarray,
    fraction: float,
):
    flat = probability.ravel()

    n = max(
        1,
        int(
            round(
                flat.size
                * fraction
            )
        ),
    )

    idx = np.argpartition(
        flat,
        -n,
    )[-n:]

    mask = np.zeros(
        flat.shape,
        dtype=bool,
    )

    mask[idx] = True

    return mask.reshape(
        probability.shape
    )


def mask_distance(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    if (
        not source.any()
        or not target.any()
    ):
        return -1.0

    target_dist = ndimage.distance_transform_edt(
        ~target
    )

    values = target_dist[
        source
    ]

    if values.size == 0:
        return -1.0

    return float(
        values.mean()
    )


# ============================================================================
# CASE LOADING
# ============================================================================
def load_case(
    row: pd.Series,
    part9: Any,
    part11: Any,
):
    loaded = part11.load_tensor_case(
        row,
        part9,
    )

    if (
        not isinstance(
            loaded,
            (tuple, list),
        )
        or len(loaded) < 2
    ):
        raise RuntimeError(
            "Unexpected Part 11 load_tensor_case() result."
        )

    image = torch.as_tensor(
        loaded[0]
    )

    mask = torch.as_tensor(
        loaded[1]
    )

    if (
        image.ndim == 4
        and image.shape[0] == 1
    ):
        image = image.squeeze(0)

    if (
        mask.ndim == 4
        and mask.shape[0] == 1
    ):
        mask = mask.squeeze(0)

    image_np = (
        image
        .detach()
        .cpu()
        .numpy()
    )

    mask_np = (
        mask
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )

    resized_mask = part11.resize_3d(
        mask_np,
        FULL,
        is_mask=True,
    )

    r2 = dilate_labels(
        resized_mask,
        2,
    )

    q = np.argwhere(
        r2 > 0
    )

    if len(q):
        center = q.mean(
            axis=0
        )
    else:
        center = np.array(
            [31.5, 47.5, 47.5]
        )

    _, starts = centered_crop(
        r2,
        center,
        CROP,
    )

    z0, y0, x0 = starts

    image_resized = part11.resize_3d(
        image_np,
        FULL,
        is_mask=False,
    )

    image_crop = image_resized[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    mask_crop = r2[
        z0:z0 + CROP[0],
        y0:y0 + CROP[1],
        x0:x0 + CROP[2],
    ]

    return (
        torch.as_tensor(
            image_crop,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            mask_crop,
            dtype=torch.long,
        ),
        starts,
        r2,
    )


# ============================================================================
# ANNOTATION COORDINATE NORMALIZATION
# ============================================================================
def annotation_points_for_case(
    row: pd.Series,
    starts: tuple[int, int, int],
    original_shape: tuple[int, int, int],
):
    """
    Try to recover point annotations associated with the current case.

    Coordinates are interpreted conservatively:
    - if already in final crop coordinates, retain them
    - if in full preprocessed coordinates, scale/translate into crop
    - otherwise, if dimensions indicate original image coordinates,
      scale to FULL and then translate into crop

    No point is invented when the source representation is unavailable.
    """

    ann = extract_annotation_table(
        row
    )

    # If the cohort has no direct annotations, look for a case-level file.
    if ann.empty:
        p = find_annotation_file(
            row
        )

        if p is not None:
            try:
                ann = normalize_annotation_dataframe(
                    pd.read_csv(p)
                )
            except Exception:
                ann = pd.DataFrame()

    if ann.empty:
        return []

    result = []

    for _, a in ann.iterrows():
        cls = map_annotation_class(
            a.get("class")
        )

        if cls not in FOCUS_CLASSES:
            continue

        point = np.array(
            [
                float(a["z"]),
                float(a["y"]),
                float(a["x"]),
            ],
            dtype=float,
        )

        # Case where coordinates already look like final crop coordinates.
        if point_inside(
            point,
            CROP,
        ):
            crop_point = point

        # Full preprocessed coordinates.
        elif point_inside(
            point,
            FULL,
        ):
            crop_point = point - np.asarray(
                starts,
                dtype=float,
            )

        else:
            # Treat as original-image coordinates if original shape is
            # available. Scale to FULL first, then apply crop origin.
            scaled = (
                point
                * (
                    np.asarray(
                        FULL,
                        dtype=float,
                    )
                    / np.asarray(
                        original_shape,
                        dtype=float,
                    )
                )
            )

            crop_point = (
                scaled
                - np.asarray(
                    starts,
                    dtype=float,
                )
            )

        result.append({
            "class_id": cls,
            "point": crop_point,
        })

    return result


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    banner(
        "PART 75"
    )

    banner(
        "PART 75 SUPERVISION + ANATOMICAL LOCALIZATION AUDIT"
    )

    print(
        "Evaluation only."
    )
    print(
        "No training."
    )
    print(
        "No optimizer."
    )
    print(
        "No SPIDER."
    )
    print(
        "No RSNA test set."
    )

    banner(
        "PART 75 PATH VALIDATION"
    )

    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 validation cohort": VAL_CSV,
        "Part 71 class-balanced E3": CKPT,
    }

    missing = []

    for name, path in required.items():
        ok = path.exists()

        print(
            f"{name:<34}: "
            f"{'FOUND' if ok else 'MISSING'}"
        )

        if not ok:
            missing.append(
                str(path)
            )

    if missing:
        raise FileNotFoundError(
            "\n".join(
                missing
            )
        )

    val_rows = pd.read_csv(
        VAL_CSV
    ).head(
        VAL_N
    ).copy()

    if len(val_rows) != VAL_N:
        raise RuntimeError(
            f"Expected {VAL_N} validation cases, "
            f"found {len(val_rows)}."
        )

    print(
        f"Device              : {DEVICE}"
    )
    print(
        f"Validation cases    : {VAL_N}"
    )
    print(
        f"Full shape          : {FULL}"
    )
    print(
        f"Crop shape          : {CROP}"
    )
    print(
        "Supervision         : R2_FULL"
    )
    print(
        "Checkpoint          : Part 71 class-balanced E3"
    )
    print(
        "Focus classes       : C2, C3"
    )
    print(
        "Best thresholds     : C2=0.15, C3=0.20"
    )

    part11 = load_module(
        P11,
        "part75_part11",
    )

    part9 = load_module(
        P9,
        "part75_part9",
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
    )

    state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model = part11.create_model(
        DEVICE
    ).to(
        DEVICE
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    case_records = []
    annotation_records = []
    top_records = []

    annotation_cases = 0
    annotation_points_total = 0

    # ========================================================================
    # CASE LOOP
    # ========================================================================
    with torch.no_grad():
        for case_no, (_, row) in enumerate(
            val_rows.iterrows(),
            start=1,
        ):
            image, target, starts, r2_full = load_case(
                row,
                part9,
                part11,
            )

            x = (
                image
                .unsqueeze(0)
                .unsqueeze(0)
                .to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type="cuda",
                enabled=DEVICE.type == "cuda",
            ):
                logits = model(
                    x
                )

            probs = torch.softmax(
                logits.float(),
                dim=1,
            )[0].cpu().numpy()

            target_np = target.numpy()

            # Original mask shape is useful if annotation coordinates are
            # stored in source-image space.
            original_shape = tuple(
                int(v)
                for v in r2_full.shape
            )

            annotations = annotation_points_for_case(
                row,
                starts,
                original_shape,
            )

            if annotations:
                annotation_cases += 1
                annotation_points_total += len(
                    annotations
                )

            # --------------------------------------------------------------
            # Case-level C2/C3 audit
            # --------------------------------------------------------------
            for c in FOCUS_CLASSES:
                probability = probs[c]
                truth = (
                    target_np == c
                )

                threshold = BEST_THRESHOLDS[c]

                pred = (
                    probability
                    >= threshold
                )

                pred_centroid = centroid(
                    pred
                )

                target_centroid = centroid(
                    truth
                )

                pred_bbox = bbox(
                    pred
                )

                target_bbox = bbox(
                    truth
                )

                pred_target_distance = mask_distance(
                    pred,
                    truth,
                )

                target_pred_distance = mask_distance(
                    truth,
                    pred,
                )

                # High-probability top 1%.
                top1 = top_probability_mask(
                    probability,
                    0.01,
                )

                top1_centroid = centroid(
                    top1
                )

                top1_to_target = mask_distance(
                    top1,
                    truth,
                )

                target_to_top1 = mask_distance(
                    truth,
                    top1,
                )

                # Probability-weighted centroid.
                prob_centroid = (
                    probability_weighted_centroid(
                        probability
                    )
                )

                target_n = int(
                    truth.sum()
                )

                pred_n = int(
                    pred.sum()
                )

                tp = int(
                    np.logical_and(
                        pred,
                        truth,
                    ).sum()
                )

                fp = int(
                    np.logical_and(
                        pred,
                        ~truth,
                    ).sum()
                )

                fn = int(
                    np.logical_and(
                        ~pred,
                        truth,
                    ).sum()
                )

                dice = safe_div(
                    2 * tp,
                    pred_n + target_n,
                )

                precision = safe_div(
                    tp,
                    tp + fp,
                )

                recall = safe_div(
                    tp,
                    tp + fn,
                )

                # Annotation-specific distances.
                class_points = [
                    np.asarray(
                        a["point"],
                        dtype=float,
                    )
                    for a in annotations
                    if a["class_id"] == c
                ]

                if class_points:
                    point_array = np.vstack(
                        class_points
                    )

                    inside_points = [
                        p
                        for p in point_array
                        if point_inside(
                            p,
                            CROP,
                        )
                    ]

                    point_target_distances = [
                        nearest_distance_to_mask(
                            p,
                            truth,
                        )
                        for p in point_array
                    ]

                    point_pred_distances = [
                        nearest_distance_to_mask(
                            p,
                            pred,
                        )
                        for p in point_array
                    ]

                    point_top1_distances = [
                        nearest_distance_to_mask(
                            p,
                            top1,
                        )
                        for p in point_array
                    ]

                    mean_point_target_distance = (
                        float(
                            np.mean(
                                point_target_distances
                            )
                        )
                        if point_target_distances
                        else -1.0
                    )

                    mean_point_pred_distance = (
                        float(
                            np.mean(
                                point_pred_distances
                            )
                        )
                        if point_pred_distances
                        else -1.0
                    )

                    mean_point_top1_distance = (
                        float(
                            np.mean(
                                point_top1_distances
                            )
                        )
                        if point_top1_distances
                        else -1.0
                    )

                    inside_count = len(
                        inside_points
                    )

                    point_count = len(
                        point_array
                    )

                else:
                    point_count = 0
                    inside_count = 0
                    mean_point_target_distance = -1.0
                    mean_point_pred_distance = -1.0
                    mean_point_top1_distance = -1.0

                case_records.append({
                    "case_no": case_no,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "threshold": threshold,
                    "target_voxels": target_n,
                    "predicted_voxels": pred_n,
                    "prediction_target_ratio":
                        safe_div(
                            pred_n,
                            target_n,
                        ),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "dice": dice,
                    "precision": precision,
                    "recall": recall,
                    "target_centroid_z":
                        (
                            target_centroid[0]
                            if target_centroid
                            else -1.0
                        ),
                    "target_centroid_y":
                        (
                            target_centroid[1]
                            if target_centroid
                            else -1.0
                        ),
                    "target_centroid_x":
                        (
                            target_centroid[2]
                            if target_centroid
                            else -1.0
                        ),
                    "prediction_centroid_z":
                        (
                            pred_centroid[0]
                            if pred_centroid
                            else -1.0
                        ),
                    "prediction_centroid_y":
                        (
                            pred_centroid[1]
                            if pred_centroid
                            else -1.0
                        ),
                    "prediction_centroid_x":
                        (
                            pred_centroid[2]
                            if pred_centroid
                            else -1.0
                        ),
                    "probability_centroid_z":
                        (
                            prob_centroid[0]
                            if prob_centroid
                            else -1.0
                        ),
                    "probability_centroid_y":
                        (
                            prob_centroid[1]
                            if prob_centroid
                            else -1.0
                        ),
                    "probability_centroid_x":
                        (
                            prob_centroid[2]
                            if prob_centroid
                            else -1.0
                        ),
                    "top1_centroid_z":
                        (
                            top1_centroid[0]
                            if top1_centroid
                            else -1.0
                        ),
                    "top1_centroid_y":
                        (
                            top1_centroid[1]
                            if top1_centroid
                            else -1.0
                        ),
                    "top1_centroid_x":
                        (
                            top1_centroid[2]
                            if top1_centroid
                            else -1.0
                        ),
                    "prediction_to_target_distance":
                        pred_target_distance,
                    "target_to_prediction_distance":
                        target_pred_distance,
                    "top1_to_target_distance":
                        top1_to_target,
                    "target_to_top1_distance":
                        target_to_top1,
                    "annotation_count":
                        point_count,
                    "annotation_inside_crop":
                        inside_count,
                    "annotation_target_distance":
                        mean_point_target_distance,
                    "annotation_prediction_distance":
                        mean_point_pred_distance,
                    "annotation_top1_distance":
                        mean_point_top1_distance,
                })

                # ----------------------------------------------------------
                # Top probability localization at 0.1%, 0.5%, 1%
                # ----------------------------------------------------------
                for fraction in TOP_FRACTIONS:
                    top = top_probability_mask(
                        probability,
                        fraction,
                    )

                    top_records.append({
                        "case_no": case_no,
                        "class_id": c,
                        "class_name": CLASSES[c],
                        "top_fraction": fraction,
                        "top_voxels":
                            int(top.sum()),
                        "target_voxels":
                            target_n,
                        "target_recall":
                            safe_div(
                                np.logical_and(
                                    top,
                                    truth,
                                ).sum(),
                                target_n,
                            ),
                        "target_precision":
                            safe_div(
                                np.logical_and(
                                    top,
                                    truth,
                                ).sum(),
                                int(top.sum()),
                            ),
                        "top_to_target_distance":
                            mask_distance(
                                top,
                                truth,
                            ),
                        "target_to_top_distance":
                            mask_distance(
                                truth,
                                top,
                            ),
                    })

            # --------------------------------------------------------------
            # Store annotation-level rows.
            # --------------------------------------------------------------
            for ann_idx, ann in enumerate(
                annotations,
                start=1,
            ):
                c = ann["class_id"]
                p = np.asarray(
                    ann["point"],
                    dtype=float,
                )

                truth = (
                    target_np == c
                )

                pred = (
                    probs[c]
                    >= BEST_THRESHOLDS[c]
                )

                top1 = top_probability_mask(
                    probs[c],
                    0.01,
                )

                annotation_records.append({
                    "case_no": case_no,
                    "annotation_index": ann_idx,
                    "class_id": c,
                    "class_name": CLASSES[c],
                    "point_z":
                        float(p[0]),
                    "point_y":
                        float(p[1]),
                    "point_x":
                        float(p[2]),
                    "inside_crop":
                        point_inside(
                            p,
                            CROP,
                        ),
                    "distance_to_target":
                        nearest_distance_to_mask(
                            p,
                            truth,
                        ),
                    "distance_to_threshold_prediction":
                        nearest_distance_to_mask(
                            p,
                            pred,
                        ),
                    "distance_to_top1_prediction":
                        nearest_distance_to_mask(
                            p,
                            top1,
                        ),
                    "point_on_target":
                        bool(
                            point_inside(
                                np.round(p).astype(
                                    int
                                ),
                                CROP,
                            )
                            and truth[
                                tuple(
                                    np.clip(
                                        np.round(p).astype(
                                            int
                                        ),
                                        0,
                                        np.asarray(
                                            CROP
                                        ) - 1,
                                    )
                                )
                            ]
                        ),
                })

    # ========================================================================
    # DATAFRAMES
    # ========================================================================
    case_df = pd.DataFrame(
        case_records
    )

    top_df = pd.DataFrame(
        top_records
    )

    annotation_df = pd.DataFrame(
        annotation_records
    )

    # ========================================================================
    # AGGREGATE CASE SUMMARY
    # ========================================================================
    summary_rows = []

    for c in FOCUS_CLASSES:
        g = case_df[
            case_df.class_id == c
        ]

        summary_rows.append({
            "class_id": c,
            "class_name": CLASSES[c],
            "cases": len(g),
            "mean_target_voxels":
                float(
                    g.target_voxels.mean()
                ),
            "mean_predicted_voxels":
                float(
                    g.predicted_voxels.mean()
                ),
            "mean_prediction_target_ratio":
                float(
                    g.prediction_target_ratio.mean()
                ),
            "mean_dice":
                float(
                    g.dice.mean()
                ),
            "mean_precision":
                float(
                    g.precision.mean()
                ),
            "mean_recall":
                float(
                    g.recall.mean()
                ),
            "mean_prediction_to_target_distance":
                float(
                    g[
                        g.prediction_to_target_distance
                        >= 0
                    ].prediction_to_target_distance.mean()
                ),
            "mean_target_to_prediction_distance":
                float(
                    g[
                        g.target_to_prediction_distance
                        >= 0
                    ].target_to_prediction_distance.mean()
                ),
            "mean_top1_to_target_distance":
                float(
                    g[
                        g.top1_to_target_distance
                        >= 0
                    ].top1_to_target_distance.mean()
                ),
            "mean_target_to_top1_distance":
                float(
                    g[
                        g.target_to_top1_distance
                        >= 0
                    ].target_to_top1_distance.mean()
                ),
            "mean_annotation_target_distance":
                float(
                    g[
                        g.annotation_target_distance
                        >= 0
                    ].annotation_target_distance.mean()
                ),
            "mean_annotation_prediction_distance":
                float(
                    g[
                        g.annotation_prediction_distance
                        >= 0
                    ].annotation_prediction_distance.mean()
                ),
            "mean_annotation_top1_distance":
                float(
                    g[
                        g.annotation_top1_distance
                        >= 0
                    ].annotation_top1_distance.mean()
                ),
            "cases_with_annotations":
                int(
                    (
                        g.annotation_count
                        > 0
                    ).sum()
                ),
            "mean_annotations_per_case":
                float(
                    g.annotation_count.mean()
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================================
    # TOP PROBABILITY SUMMARY
    # ========================================================================
    top_summary_rows = []

    for c in FOCUS_CLASSES:
        for fraction in TOP_FRACTIONS:
            g = top_df[
                (top_df.class_id == c)
                & (
                    top_df.top_fraction
                    == fraction
                )
            ]

            top_summary_rows.append({
                "class_id": c,
                "class_name": CLASSES[c],
                "top_fraction": fraction,
                "mean_target_recall":
                    float(
                        g.target_recall.mean()
                    ),
                "mean_target_precision":
                    float(
                        g.target_precision.mean()
                    ),
                "mean_top_to_target_distance":
                    float(
                        g[
                            g.top_to_target_distance
                            >= 0
                        ].top_to_target_distance.mean()
                    ),
                "mean_target_to_top_distance":
                    float(
                        g[
                            g.target_to_top_distance
                            >= 0
                        ].target_to_top_distance.mean()
                    ),
            })

    top_summary_df = pd.DataFrame(
        top_summary_rows
    )

    # ========================================================================
    # ANNOTATION SUMMARY
    # ========================================================================
    if not annotation_df.empty:
        annotation_summary_rows = []

        for c in FOCUS_CLASSES:
            g = annotation_df[
                annotation_df.class_id == c
            ]

            annotation_summary_rows.append({
                "class_id": c,
                "class_name": CLASSES[c],
                "annotations": len(g),
                "inside_crop_fraction":
                    float(
                        g.inside_crop.mean()
                    ),
                "point_on_target_fraction":
                    float(
                        g.point_on_target.mean()
                    ),
                "mean_point_target_distance":
                    float(
                        g.distance_to_target.mean()
                    ),
                "mean_point_threshold_prediction_distance":
                    float(
                        g[
                            g.distance_to_threshold_prediction
                            >= 0
                        ].distance_to_threshold_prediction.mean()
                    ),
                "mean_point_top1_prediction_distance":
                    float(
                        g[
                            g.distance_to_top1_prediction
                            >= 0
                        ].distance_to_top1_prediction.mean()
                    ),
            })

        annotation_summary_df = pd.DataFrame(
            annotation_summary_rows
        )
    else:
        annotation_summary_df = pd.DataFrame(
            columns=[
                "class_id",
                "class_name",
                "annotations",
            ]
        )

    # ========================================================================
    # DIAGNOSTIC INTERPRETATION
    # ========================================================================
    c2 = summary_df[
        summary_df.class_id == 2
    ].iloc[0]

    c3 = summary_df[
        summary_df.class_id == 3
    ].iloc[0]

    mean_ratio = float(
        (
            c2[
                "mean_prediction_target_ratio"
            ]
            + c3[
                "mean_prediction_target_ratio"
            ]
        )
        / 2.0
    )

    mean_pred_target_distance = float(
        (
            c2[
                "mean_prediction_to_target_distance"
            ]
            + c3[
                "mean_prediction_to_target_distance"
            ]
        )
        / 2.0
    )

    mean_target_pred_distance = float(
        (
            c2[
                "mean_target_to_prediction_distance"
            ]
            + c3[
                "mean_target_to_prediction_distance"
            ]
        )
        / 2.0
    )

    mean_top1_target_distance = float(
        (
            c2[
                "mean_top1_to_target_distance"
            ]
            + c3[
                "mean_top1_to_target_distance"
            ]
        )
        / 2.0
    )

    if (
        mean_ratio >= 10.0
        and mean_top1_target_distance >= 5.0
    ):
        diagnosis = (
            "C2_C3_SUPERVISION_OR_LOCALIZATION_QUALITY_IS_THE_DOMINANT_REMAINING_BOTTLENECK"
        )
    elif mean_ratio >= 10.0:
        diagnosis = (
            "C2_C3_PREDICTIONS_ARE_STRONGLY_OVERSIZED_RELATIVE_TO_SUPERVISION"
        )
    elif mean_pred_target_distance >= 5.0:
        diagnosis = (
            "C2_C3_HIGH_PROBABILITY_ACTIVATION_IS_SPATIALLY_MISALIGNED"
        )
    else:
        diagnosis = (
            "C2_C3_SHOW_PARTIAL_ANATOMICAL_LOCALIZATION_BUT_REMAIN_INACCURATE"
        )

    # ========================================================================
    # PRINT
    # ========================================================================
    banner(
        "PART 75 C2/C3 SUPERVISION SUMMARY"
    )

    print(
        f"{'Class':<42}"
        f"{'Target':>10}"
        f"{'Pred':>12}"
        f"{'Ratio':>10}"
        f"{'Dice':>10}"
        f"{'P→T':>10}"
        f"{'T→P':>10}"
    )

    for _, r in summary_df.iterrows():
        print(
            f"C{int(r['class_id'])} "
            f"{r['class_name']:<36}"
            f"{r['mean_target_voxels']:>10.1f}"
            f"{r['mean_predicted_voxels']:>12.1f}"
            f"{r['mean_prediction_target_ratio']:>10.3f}"
            f"{r['mean_dice']:>10.6f}"
            f"{r['mean_prediction_to_target_distance']:>10.3f}"
            f"{r['mean_target_to_prediction_distance']:>10.3f}"
        )

    banner(
        "PART 75 TOP-PROBABILITY LOCALIZATION"
    )

    for _, r in top_summary_df[
        top_summary_df.top_fraction == 0.01
    ].iterrows():
        print(
            f"C{int(r['class_id'])} "
            f"{r['class_name']}:"
        )
        print(
            f"  Top 1% target recall : "
            f"{r['mean_target_recall']:.6f}"
        )
        print(
            f"  Top 1% precision     : "
            f"{r['mean_target_precision']:.6f}"
        )
        print(
            f"  Top 1% → target dist : "
            f"{r['mean_top_to_target_distance']:.3f}"
        )
        print(
            f"  Target → top 1% dist : "
            f"{r['mean_target_to_top_distance']:.3f}"
        )

    banner(
        "PART 75 RSNA POINT / PSEUDOMASK / PREDICTION AUDIT"
    )

    print(
        f"Cases with usable annotations : "
        f"{annotation_cases}/{VAL_N}"
    )

    print(
        f"Total usable annotations       : "
        f"{annotation_points_total}"
    )

    if not annotation_summary_df.empty:
        for _, r in annotation_summary_df.iterrows():
            print(
                f"C{int(r['class_id'])} "
                f"{r['class_name']}:"
            )
            print(
                f"  Annotations                 : "
                f"{int(r['annotations'])}"
            )
            print(
                f"  Inside crop fraction        : "
                f"{r['inside_crop_fraction']:.6f}"
            )
            print(
                f"  Point on target fraction    : "
                f"{r['point_on_target_fraction']:.6f}"
            )
            print(
                f"  Point → target distance     : "
                f"{r['mean_point_target_distance']:.3f}"
            )
            print(
                f"  Point → prediction distance: "
                f"{r['mean_point_threshold_prediction_distance']:.3f}"
            )
            print(
                f"  Point → top1 distance       : "
                f"{r['mean_point_top1_prediction_distance']:.3f}"
            )

    banner(
        "PART 75 FINAL DIAGNOSIS"
    )

    print(
        f"C2/C3 mean prediction/target ratio : "
        f"{mean_ratio:.3f}"
    )

    print(
        f"C2/C3 mean prediction→target dist  : "
        f"{mean_pred_target_distance:.3f}"
    )

    print(
        f"C2/C3 mean target→prediction dist  : "
        f"{mean_target_pred_distance:.3f}"
    )

    print(
        f"C2/C3 mean top1→target distance    : "
        f"{mean_top1_target_distance:.3f}"
    )

    print(
        f"Diagnosis                           : "
        f"{diagnosis}"
    )

    # ========================================================================
    # SAVE
    # ========================================================================
    case_df.to_csv(
        REPORT
        / "part75_case_c2_c3_localization_metrics.csv",
        index=False,
    )

    summary_df.to_csv(
        REPORT
        / "part75_c2_c3_localization_summary.csv",
        index=False,
    )

    top_df.to_csv(
        REPORT
        / "part75_case_top_probability_metrics.csv",
        index=False,
    )

    top_summary_df.to_csv(
        REPORT
        / "part75_top_probability_summary.csv",
        index=False,
    )

    annotation_df.to_csv(
        REPORT
        / "part75_annotation_level_audit.csv",
        index=False,
    )

    annotation_summary_df.to_csv(
        REPORT
        / "part75_annotation_summary.csv",
        index=False,
    )

    summary = {
        "part": 75,
        "validation_cases": VAL_N,
        "full_shape": FULL,
        "crop_shape": CROP,
        "checkpoint":
            "part71_r2_full_class_balanced_epoch3.pth",
        "focus_classes": FOCUS_CLASSES,
        "best_thresholds": BEST_THRESHOLDS,
        "mean_c2_c3_prediction_target_ratio":
            mean_ratio,
        "mean_c2_c3_prediction_to_target_distance":
            mean_pred_target_distance,
        "mean_c2_c3_target_to_prediction_distance":
            mean_target_pred_distance,
        "mean_c2_c3_top1_to_target_distance":
            mean_top1_target_distance,
        "annotation_cases":
            annotation_cases,
        "annotation_points_total":
            annotation_points_total,
        "diagnosis":
            diagnosis,
    }

    with (
        REPORT
        / "part75_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    with (
        REPORT
        / "part75_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "PART 75\n"
            "RSNA-ONLY C2/C3 SUPERVISION + "
            "ANATOMICAL LOCALIZATION AUDIT\n\n"
        )

        f.write(
            f"C2/C3 mean prediction-target ratio: "
            f"{mean_ratio:.6f}\n"
        )

        f.write(
            f"C2/C3 mean prediction-to-target distance: "
            f"{mean_pred_target_distance:.6f}\n"
        )

        f.write(
            f"C2/C3 mean target-to-prediction distance: "
            f"{mean_target_pred_distance:.6f}\n"
        )

        f.write(
            f"C2/C3 mean top1-to-target distance: "
            f"{mean_top1_target_distance:.6f}\n"
        )

        f.write(
            f"Usable annotation cases: "
            f"{annotation_cases}/{VAL_N}\n"
        )

        f.write(
            f"Usable annotation points: "
            f"{annotation_points_total}\n"
        )

        f.write(
            f"Diagnosis: {diagnosis}\n\n"
        )

        f.write(
            summary_df.to_string(
                index=False
            )
        )

    banner(
        "PART 75 OUTPUTS"
    )

    for p in sorted(
        REPORT.iterdir()
    ):
        print(p)

    del model
    reset_cuda()


if __name__ == "__main__":
    main()
