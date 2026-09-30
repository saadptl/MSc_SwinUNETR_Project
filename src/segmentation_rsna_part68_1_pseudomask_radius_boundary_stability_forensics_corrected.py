"""
PART 68.1 — CORRECTED PSEUDO-MASK RADIUS / BOUNDARY STABILITY FORENSICS

Purpose
-------
Corrected version of Part 68.

Part 68's radius/voxel statistics were valid, but its annotation extraction
returned zero annotations. Therefore its annotation-agreement diagnosis was
invalid.

Part 68.1 keeps the valid radius analysis and reuses the EXACT Part 67
annotation mapping/crop logic:

    - first 100 Part-15 validation cases
    - RSNA train_label_coordinates.csv
    - Part 11 robust DICOM loading
    - native -> FULL=(64,96,96) coordinate mapping
    - R2 pseudo-mask construction
    - foreground-centered 32x64x64 crop
    - exact point inclusion / rounding handling
    - exact local same-class neighborhood logic

It compares:
    R0, R1, R2, R3
    R2_CORE
    R2_HALO

No training, optimizer, backward pass, checkpoint modification, SPIDER,
or RSNA test set is used.

Important scientific limitation
-------------------------------
RSNA coordinates are point annotations, not manual segmentation masks.
Point agreement is therefore a target-consistency diagnostic, NOT medical
segmentation accuracy.

The R2_CORE/R2_HALO split is a morphological sensitivity analysis only.
It does not establish that either region is medically correct.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(
    r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project"
)

P11 = ROOT / "src" / "segmentation_rsna_part11_controlled_pilot_training.py"
P9 = ROOT / "src" / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
)

VACSV = P15 / "part15_validation_cohort.csv"

RSNA = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

COORD = RSNA / "train_label_coordinates.csv"

OUT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part68_1_pseudomask_radius_boundary_stability_forensics_corrected"
)

REPORT = OUT / "reports"


# ============================================================================
# LOCKED CONFIGURATION
# ============================================================================

N = 100

FULL = (64, 96, 96)
CROP = (32, 64, 64)

RADII = (0, 1, 2, 3)

# R2 core/halo:
#   R2_CORE = erosion of binary R2 foreground by one voxel
#   R2_HALO = R2 foreground minus R2_CORE
#
# This is intentionally a sensitivity partition, not a medical claim.
CORE_EROSION_ITERATIONS = 1

CLASSES = {
    1: "Spinal_Canal_Stenosis",
    2: "Left_Neural_Foraminal_Narrowing",
    3: "Right_Neural_Foraminal_Narrowing",
    4: "Left_Subarticular_Stenosis",
    5: "Right_Subarticular_Stenosis",
}

NAME = {v: k for k, v in CLASSES.items()}
NAME.update(
    {
        "Spinal Canal Stenosis": 1,
        "Left Neural Foraminal Narrowing": 2,
        "Right Neural Foraminal Narrowing": 3,
        "Left Subarticular Stenosis": 4,
        "Right Subarticular Stenosis": 5,
    }
)


# ============================================================================
# UTILITIES
# ============================================================================

def banner(s: str) -> None:
    print("\n" + "=" * 88)
    print(s)
    print("=" * 88)


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_paths() -> None:
    required = {
        "Project root": ROOT,
        "Part 11": P11,
        "Part 9": P9,
        "Part 15 validation cohort": VACSV,
        "RSNA coordinate CSV": COORD,
    }

    banner("PART 68.1 PATH VALIDATION")

    missing = []

    for name, path in required.items():
        exists = path.exists()
        print(f"{name:<36}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Required input(s) missing:\n" + "\n".join(missing)
        )


# ============================================================================
# EXACT PART 67 MORPHOLOGY / POINT HELPERS
# ============================================================================

def dil6(a: np.ndarray, n: int) -> np.ndarray:
    """Repeated 3-D 6-connected binary dilation."""
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


def dilate_labels(m: np.ndarray, n: int) -> np.ndarray:
    """
    Exact Part 67 label dilation logic.

    Each foreground class is dilated independently. Overlaps are resolved
    deterministically by larger original class voxel count, then class ID.
    """
    if n == 0:
        return m.astype(np.int64).copy()

    parts = []

    for c in range(1, 6):
        q = m == c

        if q.any():
            parts.append(
                (
                    int(q.sum()),
                    c,
                    dil6(q, n),
                )
            )

    parts.sort(key=lambda x: (-x[0], x[1]))

    out = np.zeros_like(m, dtype=np.int64)
    used = np.zeros_like(m, dtype=bool)

    for _, c, q in parts:
        q = q & ~used
        out[q] = c
        used |= q

    return out


def map_coord(x: float, y: float, z: float, native) -> tuple[float, float, float]:
    """Exact Part 67 native -> FULL coordinate mapping."""
    nz, nh, nw = map(float, native)

    return (
        (z + 0.5) * FULL[0] / nz - 0.5,
        (y + 0.5) * FULL[1] / nh - 0.5,
        (x + 0.5) * FULL[2] / nw - 0.5,
    )


def crop_mask(
    m: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Exact Part 67 foreground-centered crop."""
    st = [
        max(
            0,
            min(
                int(round(center[i])) - CROP[i] // 2,
                FULL[i] - CROP[i],
            ),
        )
        for i in range(3)
    ]

    z, y, x = st
    dz, dy, dx = CROP

    return (
        m[z:z + dz, y:y + dy, x:x + dx],
        tuple(st),
    )


def offsets(r: int):
    """Exact Part 67 Manhattan-ball offsets."""
    return [
        (a, b, c)
        for a in range(-r, r + 1)
        for b in range(-r, r + 1)
        for c in range(-r, r + 1)
        if abs(a) + abs(b) + abs(c) <= r
    ]


def local_label_fractions(
    m: np.ndarray,
    z: int,
    y: int,
    x: int,
    c: int,
    r: int,
) -> tuple[float, float]:
    """Exact Part 67 local same-class and foreground fractions."""
    vals = []
    correct = 0
    fg = 0

    for a, b, d in offsets(r):
        zz, yy, xx = z + a, y + b, x + d

        if (
            0 <= zz < m.shape[0]
            and 0 <= yy < m.shape[1]
            and 0 <= xx < m.shape[2]
        ):
            v = int(m[zz, yy, xx])
            vals.append(v)
            fg += int(v > 0)
            correct += int(v == c)

    n = len(vals)

    return (
        correct / n if n else 0.0,
        fg / n if n else 0.0,
    )


def nearest_dist(
    m: np.ndarray,
    z: int,
    y: int,
    x: int,
    c: int | None,
    maxd: int = 20,
) -> float:
    """Exact Part 67 Manhattan nearest-distance search."""
    target = (m > 0) if c is None else (m == c)

    if (
        0 <= z < m.shape[0]
        and 0 <= y < m.shape[1]
        and 0 <= x < m.shape[2]
        and target[z, y, x]
    ):
        return 0.0

    for d in range(1, maxd + 1):
        for a, b, cc in offsets(d):
            if abs(a) + abs(b) + abs(cc) != d:
                continue

            zz, yy, xx = z + a, y + b, x + cc

            if (
                0 <= zz < target.shape[0]
                and 0 <= yy < target.shape[1]
                and 0 <= xx < target.shape[2]
                and target[zz, yy, xx]
            ):
                return float(d)

    return float("inf")


# ============================================================================
# CASE LOADING
# ============================================================================

def load_case(
    p11,
    p9,
    row: pd.Series,
    coord: pd.DataFrame,
):
    """
    Load one case and reproduce the Part 67 coordinate pipeline.

    Returns:
        image_full
        mask_r2_full
        annotation_points
        native_shape
    """
    im, m, _ = p11.load_case_robust(row, p9)

    im = np.asarray(im, np.float32)
    m = np.asarray(m, np.int64)

    native = m.shape

    im = p11.resize_3d(
        im,
        FULL,
        is_mask=False,
    )

    # Exact Part 67 R2 pseudo-mask.
    m_r2 = dilate_labels(
        p11.resize_3d(
            m,
            FULL,
            is_mask=True,
        ),
        2,
    )

    sid = str(row["study_id"])
    ser = str(row["series_id"])

    ann = coord[
        (coord.study_id.astype(str) == sid)
        & (coord.series_id.astype(str) == ser)
    ]

    _, ds, _ = p11.read_dicom_series_robust(
        p11.resolve_series_dir(row)
    )

    inst_to_z = {
        int(d.get("InstanceNumber", 0)): i
        for i, d in enumerate(ds)
    }

    points = []

    for _, a in ann.iterrows():

        cid = NAME.get(
            str(a["condition"]).strip()
        )

        try:
            x = float(a["x"])
            y = float(a["y"])
            inst = int(float(a["instance_number"]))
        except Exception:
            continue

        if cid is None or inst not in inst_to_z:
            continue

        z = inst_to_z[inst]

        if not (
            0 <= x < native[2]
            and 0 <= y < native[1]
            and 0 <= z < native[0]
        ):
            continue

        zz, yy, xx = map_coord(
            x,
            y,
            z,
            native,
        )

        points.append(
            {
                "class_id": cid,
                "class_name": CLASSES[cid],
                "z": zz,
                "y": yy,
                "x": xx,
                "level": str(a.get("level", "")),
                "condition": str(a["condition"]),
                "instance_number": inst,
                "native_x": x,
                "native_y": y,
                "native_z": z,
            }
        )

    return im, m_r2, points, native


# ============================================================================
# RADIUS / CORE / HALO REGIONS
# ============================================================================

def build_regions(r2_mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build R0/R1/R2/R3 from the exact Part 67 R2 target by reconstructing
    the underlying R0 mask is not possible from R2 alone.

    Therefore this function is NOT used for R0/R1/R3.

    It exists only for the R2 core/halo split.
    """
    r2_binary = r2_mask > 0

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    core_binary = ndimage.binary_erosion(
        r2_binary,
        structure=structure,
        iterations=CORE_EROSION_ITERATIONS,
        border_value=0,
    )

    halo_binary = r2_binary & ~core_binary

    core = r2_mask.copy()
    core[~core_binary] = 0

    halo = r2_mask.copy()
    halo[~halo_binary] = 0

    return {
        "R2_CORE": core.astype(np.int64),
        "R2_HALO": halo.astype(np.int64),
    }


def make_all_regions(
    p11,
    raw_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Reconstruct R0/R1/R2/R3 from the original resized Part 11 pseudo-mask.
    """
    resized = p11.resize_3d(
        raw_mask,
        FULL,
        is_mask=True,
    )

    regions = {
        f"R{r}": dilate_labels(resized, r)
        for r in RADII
    }

    regions.update(
        build_regions(regions["R2"])
    )

    return regions


# ============================================================================
# REGION STATISTICS
# ============================================================================

def connected_component_stats(mask: np.ndarray):
    binary = mask > 0

    if not binary.any():
        return 0, 0, 0.0, 0.0

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    labels, n = ndimage.label(
        binary,
        structure=structure,
    )

    counts = np.bincount(
        labels.ravel()
    )[1:]

    if len(counts) == 0:
        return 0, 0, 0.0, 0.0

    largest = int(counts.max())
    small_fraction = float(
        (counts < 5).sum() / len(counts)
    )

    active_z = float(
        np.any(binary, axis=(1, 2)).mean()
    )

    return (
        int(n),
        largest,
        small_fraction,
        active_z,
    )


def region_record(
    split: str,
    case_no: int,
    region_name: str,
    mask: np.ndarray,
) -> dict:
    fg = int((mask > 0).sum())
    total = int(np.prod(CROP))

    components, largest, small_fraction, active_z = (
        connected_component_stats(mask)
    )

    class_counts = {
        f"class_{c}_voxels": int((mask == c).sum())
        for c in range(1, 6)
    }

    rec = {
        "split": split,
        "case_no": case_no,
        "region": region_name,
        "shape_z": mask.shape[0],
        "shape_y": mask.shape[1],
        "shape_x": mask.shape[2],
        "foreground_voxels": fg,
        "foreground_fraction": fg / total,
        "connected_components_6conn": components,
        "largest_component_voxels": largest,
        "small_component_fraction_lt5": small_fraction,
        "active_z_fraction": active_z,
    }

    rec.update(class_counts)

    return rec


# ============================================================================
# POINT AGREEMENT
# ============================================================================

def point_region_record(
    case_no: int,
    point_no: int,
    p: dict,
    region_name: str,
    crop: np.ndarray,
    st: tuple[int, int, int],
) -> dict:
    """
    Exact Part 67 point classification logic applied to a supplied region.
    """
    inside = (
        st[0] <= p["z"] < st[0] + CROP[0]
        and st[1] <= p["y"] < st[1] + CROP[1]
        and st[2] <= p["x"] < st[2] + CROP[2]
    )

    rec = {
        "case": case_no,
        "annotation_no": point_no,
        "region": region_name,
        "class_id": p["class_id"],
        "class_name": p["class_name"],
        "condition": p["condition"],
        "level": p["level"],
        "inside_crop": int(inside),
    }

    if not inside:
        rec.update(
            {
                "point_label": "OUTSIDE_CROP",
                "local_r1_same": np.nan,
                "local_r2_same": np.nan,
                "local_r3_same": np.nan,
                "nearest_same_class": np.nan,
                "nearest_any_fg": np.nan,
            }
        )

        return rec

    z, y, x = [
        int(round(p[k] - st[j]))
        for j, k in enumerate(("z", "y", "x"))
    ]

    if not all(
        0 <= v < CROP[j]
        for j, v in enumerate((z, y, x))
    ):
        rec["inside_crop"] = 0

        rec.update(
            {
                "point_label": "ROUNDING_OUTSIDE",
                "local_r1_same": np.nan,
                "local_r2_same": np.nan,
                "local_r3_same": np.nan,
                "nearest_same_class": np.nan,
                "nearest_any_fg": np.nan,
            }
        )

        return rec

    lab = int(crop[z, y, x])

    r1, _ = local_label_fractions(
        crop,
        z,
        y,
        x,
        p["class_id"],
        1,
    )

    r2, _ = local_label_fractions(
        crop,
        z,
        y,
        x,
        p["class_id"],
        2,
    )

    r3, _ = local_label_fractions(
        crop,
        z,
        y,
        x,
        p["class_id"],
        3,
    )

    point_label = (
        "CORRECT_CLASS"
        if lab == p["class_id"]
        else (
            "OTHER_FOREGROUND"
            if lab > 0
            else "BACKGROUND"
        )
    )

    rec.update(
        {
            "point_label": point_label,
            "local_r1_same": r1,
            "local_r2_same": r2,
            "local_r3_same": r3,
            "nearest_same_class": nearest_dist(
                crop,
                z,
                y,
                x,
                p["class_id"],
                20,
            ),
            "nearest_any_fg": nearest_dist(
                crop,
                z,
                y,
                x,
                None,
                20,
            ),
        }
    )

    return rec


def aggregate_point_agreement(pdf: pd.DataFrame):
    rows = []

    if pdf.empty:
        return pd.DataFrame()

    for region in pdf["region"].unique():

        g = pdf[
            pdf["region"] == region
        ]

        total = len(g)
        inside = g[g.inside_crop == 1]

        row = {
            "region": region,
            "annotations": total,
            "inside_crop": int(len(inside)),
            "inside_crop_rate": (
                float(len(inside) / total)
                if total
                else np.nan
            ),
        }

        for label in [
            "CORRECT_CLASS",
            "OTHER_FOREGROUND",
            "BACKGROUND",
            "OUTSIDE_CROP",
            "ROUNDING_OUTSIDE",
        ]:
            row[
                f"{label.lower()}_count"
            ] = int(
                (g.point_label == label).sum()
            )

        if len(inside):
            row.update(
                {
                    "correct_class_rate_inside": float(
                        (inside.point_label == "CORRECT_CLASS").mean()
                    ),
                    "background_rate_inside": float(
                        (inside.point_label == "BACKGROUND").mean()
                    ),
                    "other_foreground_rate_inside": float(
                        (inside.point_label == "OTHER_FOREGROUND").mean()
                    ),
                    "local_r1_same_mean": float(
                        inside.local_r1_same.mean()
                    ),
                    "local_r2_same_mean": float(
                        inside.local_r2_same.mean()
                    ),
                    "local_r3_same_mean": float(
                        inside.local_r3_same.mean()
                    ),
                    "local_r1_same_hit_rate": float(
                        (inside.local_r1_same > 0).mean()
                    ),
                    "local_r2_same_hit_rate": float(
                        (inside.local_r2_same > 0).mean()
                    ),
                    "local_r3_same_hit_rate": float(
                        (inside.local_r3_same > 0).mean()
                    ),
                    "nearest_same_class_mean": float(
                        inside.nearest_same_class
                        .replace(np.inf, np.nan)
                        .mean()
                    ),
                    "nearest_same_class_median": float(
                        inside.nearest_same_class
                        .replace(np.inf, np.nan)
                        .median()
                    ),
                }
            )
        else:
            for k in [
                "correct_class_rate_inside",
                "background_rate_inside",
                "other_foreground_rate_inside",
                "local_r1_same_mean",
                "local_r2_same_mean",
                "local_r3_same_mean",
                "local_r1_same_hit_rate",
                "local_r2_same_hit_rate",
                "local_r3_same_hit_rate",
                "nearest_same_class_mean",
                "nearest_same_class_median",
            ]:
                row[k] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def per_class_agreement(pdf: pd.DataFrame):
    rows = []

    if pdf.empty:
        return pd.DataFrame()

    for region in pdf["region"].unique():

        rg = pdf[
            pdf["region"] == region
        ]

        for cid, name in CLASSES.items():

            g = rg[
                rg.class_id == cid
            ]

            inside = g[
                g.inside_crop == 1
            ]

            rows.append(
                {
                    "region": region,
                    "class_id": cid,
                    "class_name": name,
                    "annotations": len(g),
                    "outside_crop_rate": (
                        float((g.inside_crop == 0).mean())
                        if len(g)
                        else np.nan
                    ),
                    "correct_class_rate_inside": (
                        float(
                            (inside.point_label == "CORRECT_CLASS").mean()
                        )
                        if len(inside)
                        else np.nan
                    ),
                    "background_rate_inside": (
                        float(
                            (inside.point_label == "BACKGROUND").mean()
                        )
                        if len(inside)
                        else np.nan
                    ),
                    "other_foreground_rate_inside": (
                        float(
                            (inside.point_label == "OTHER_FOREGROUND").mean()
                        )
                        if len(inside)
                        else np.nan
                    ),
                    "r1_mean": (
                        float(inside.local_r1_same.mean())
                        if len(inside)
                        else np.nan
                    ),
                    "r2_mean": (
                        float(inside.local_r2_same.mean())
                        if len(inside)
                        else np.nan
                    ),
                    "r3_mean": (
                        float(inside.local_r3_same.mean())
                        if len(inside)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    validate_paths()

    banner(
        "PART 68.1 — CORRECTED PSEUDO-MASK RADIUS / "
        "BOUNDARY STABILITY FORENSICS"
    )

    print(f"Validation subset        : first {N}")
    print(f"Full volume              : {FULL}")
    print(f"Crop                     : {CROP}")
    print(f"Radius comparisons       : {RADII}")
    print(
        "Core/halo definition     : "
        f"R2 core after {CORE_EROSION_ITERATIONS}-voxel "
        "6-connected erosion"
    )
    print("Training performed       : NO")
    print("Optimizer used           : NO")
    print("Backward pass            : NO")
    print("SPIDER used              : NO")
    print("RSNA test set used       : NO")
    print("Part 15 modified         : NO")
    print()
    print(
        "CORRECTION: annotation extraction follows the exact Part 67 "
        "logic; the previous Part 68 zero-annotation result is not reused."
    )

    p11 = loadmod(
        P11,
        "p11_part68_1_corrected",
    )

    p9 = loadmod(
        P9,
        "p9_part68_1_corrected",
    )

    val = pd.read_csv(
        VACSV
    ).head(N)

    coord = pd.read_csv(
        COORD
    )

    banner("PART 68.1 VALIDATION PRELOAD")

    cases = []

    total_annotations = 0

    for i, (_, row) in enumerate(
        val.iterrows(),
        start=1,
    ):

        image, r2_mask, points, native = load_case(
            p11,
            p9,
            row,
            coord,
        )

        cases.append(
            {
                "row": row,
                "image": image,
                "r2_mask": r2_mask,
                "points": points,
                "native": native,
            }
        )

        total_annotations += len(points)

        if i in (
            1,
            10,
            20,
            25,
            40,
            50,
            75,
            100,
        ):
            print(
                f"VALIDATION {i:03d}/{N} "
                f"R2_FG={int((r2_mask > 0).sum()):5d} "
                f"points={len(points):2d}"
            )

    banner("PART 68.1 ANNOTATION EXTRACTION CHECK")

    print(
        f"Mapped annotation rows : {total_annotations}"
    )

    if total_annotations == 0:
        raise RuntimeError(
            "STOP: corrected Part 68.1 still mapped zero annotations. "
            "Do not interpret any output."
        )

    print(
        "✓ Annotation extraction is non-zero."
    )

    banner("PART 68.1 RADIUS / CORE / HALO ANALYSIS")

    region_records = []
    point_records = []

    for case_no, case in enumerate(
        cases,
        start=1,
    ):

        raw_mask = case["r2_mask"]

        # IMPORTANT:
        # Part 67's R2 mask was constructed from the original mask.
        # Recovering R0 from R2 is impossible.
        #
        # Therefore reload the native mask once through Part 11 so R0/R1/R2/R3
        # are all reconstructed from the same original source.
        row = case["row"]

        _, original_mask, _ = p11.load_case_robust(
            row,
            p9,
        )

        original_mask = np.asarray(
            original_mask,
            np.int64,
        )

        regions_full = make_all_regions(
            p11,
            original_mask,
        )

        # R2 defines the single locked crop geometry, exactly as Part 67.
        r2_full = regions_full["R2"]

        q = np.argwhere(
            r2_full > 0
        )

        center = (
            q.mean(0)
            if len(q)
            else np.array(
                [31.5, 47.5, 47.5]
            )
        )

        _, st = crop_mask(
            r2_full,
            center,
        )

        # Crop all regions using the SAME R2 crop origin.
        for region_name, region_full in regions_full.items():

            z0, y0, x0 = st

            crop = region_full[
                z0:z0 + CROP[0],
                y0:y0 + CROP[1],
                x0:x0 + CROP[2],
            ]

            region_records.append(
                region_record(
                    "validation",
                    case_no,
                    region_name,
                    crop,
                )
            )

            for point_no, point in enumerate(
                case["points"],
                start=1,
            ):

                point_records.append(
                    point_region_record(
                        case_no,
                        point_no,
                        point,
                        region_name,
                        crop,
                        st,
                    )
                )

        if case_no % 10 == 0:
            print(
                f"Processed {case_no:03d}/{N} validation cases"
            )

        del regions_full
        gc.collect()

    region_df = pd.DataFrame(
        region_records
    )

    point_df = pd.DataFrame(
        point_records
    )

    agreement_df = aggregate_point_agreement(
        point_df
    )

    class_df = per_class_agreement(
        point_df
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    banner("PART 68.1 REGION SUMMARY")

    region_summary = []

    for region in [
        "R0",
        "R1",
        "R2",
        "R2_CORE",
        "R2_HALO",
        "R3",
    ]:

        g = region_df[
            region_df.region == region
        ]

        if g.empty:
            continue

        row = {
            "region": region,
            "cases": len(g),
            "mean_foreground_voxels": float(
                g.foreground_voxels.mean()
            ),
            "median_foreground_voxels": float(
                g.foreground_voxels.median()
            ),
            "mean_foreground_fraction": float(
                g.foreground_fraction.mean()
            ),
            "mean_connected_components": float(
                g.connected_components_6conn.mean()
            ),
            "mean_largest_component_voxels": float(
                g.largest_component_voxels.mean()
            ),
            "mean_small_component_fraction_lt5": float(
                g.small_component_fraction_lt5.mean()
            ),
            "mean_active_z_fraction": float(
                g.active_z_fraction.mean()
            ),
        }

        for c in range(1, 6):
            row[
                f"mean_class_{c}_voxels"
            ] = float(
                g[f"class_{c}_voxels"].mean()
            )

        region_summary.append(row)

        print(
            f"{region:<8} "
            f"FG={row['mean_foreground_voxels']:10.2f} "
            f"frac={row['mean_foreground_fraction']:.8f} "
            f"components={row['mean_connected_components']:.2f}"
        )

    region_summary_df = pd.DataFrame(
        region_summary
    )

    banner("PART 68.1 ANNOTATION AGREEMENT SUMMARY")

    for _, r in agreement_df.iterrows():

        print(
            f"{r['region']:<8} "
            f"annotations={int(r['annotations']):4d} "
            f"inside={int(r['inside_crop']):4d} "
            f"correct={r['correct_class_rate_inside']:.4f} "
            f"background={r['background_rate_inside']:.4f} "
            f"otherFG={r['other_foreground_rate_inside']:.4f} "
            f"R2hit={r['local_r2_same_hit_rate']:.4f}"
        )

    # ========================================================================
    # R2 CORE / HALO FRACTIONS
    # ========================================================================

    r2_stats = region_summary_df[
        region_summary_df.region == "R2"
    ]

    core_stats = region_summary_df[
        region_summary_df.region == "R2_CORE"
    ]

    halo_stats = region_summary_df[
        region_summary_df.region == "R2_HALO"
    ]

    if (
        not r2_stats.empty
        and not core_stats.empty
        and not halo_stats.empty
    ):
        r2_fg = float(
            r2_stats.mean_foreground_voxels.iloc[0]
        )
        core_fg = float(
            core_stats.mean_foreground_voxels.iloc[0]
        )
        halo_fg = float(
            halo_stats.mean_foreground_voxels.iloc[0]
        )

        core_fraction = (
            core_fg / r2_fg
            if r2_fg > 0
            else np.nan
        )

        halo_fraction = (
            halo_fg / r2_fg
            if r2_fg > 0
            else np.nan
        )
    else:
        r2_fg = core_fg = halo_fg = np.nan
        core_fraction = halo_fraction = np.nan

    # ========================================================================
    # DIAGNOSTIC INTERPRETATION
    # ========================================================================

    banner("PART 68.1 DIAGNOSTIC INTERPRETATION")

    overall = agreement_df[
        agreement_df.region.isin(
            ["R0", "R1", "R2", "R3"]
        )
    ]

    r2_ag = agreement_df[
        agreement_df.region == "R2"
    ]

    if r2_ag.empty:
        diagnosis = (
            "INSUFFICIENT_R2_ANNOTATION_AGREEMENT_DATA"
        )
    else:
        r2_correct = float(
            r2_ag.correct_class_rate_inside.iloc[0]
        )

        r2_inside = int(
            r2_ag.inside_crop.iloc[0]
        )

        if r2_inside == 0:
            diagnosis = (
                "INSUFFICIENT_R2_INSIDE_CROP_ANNOTATIONS"
            )
        elif r2_correct < 0.50:
            diagnosis = (
                "R2_POINT_PSEUDOMASK_ALIGNMENT_REMAINS_WEAK"
            )
        else:
            # Compare R2 core against full R2 only after the corrected
            # annotation pipeline has produced real counts.
            core_ag = agreement_df[
                agreement_df.region == "R2_CORE"
            ]

            if not core_ag.empty:
                core_correct = float(
                    core_ag.correct_class_rate_inside.iloc[0]
                )

                if abs(core_correct - r2_correct) < 0.05:
                    diagnosis = (
                        "R2_CORE_AND_FULL_R2_HAVE_SIMILAR_POINT_AGREEMENT"
                    )
                elif core_correct > r2_correct:
                    diagnosis = (
                        "R2_CORE_HAS_HIGHER_POINT_AGREEMENT_THAN_FULL_R2"
                    )
                else:
                    diagnosis = (
                        "FULL_R2_HAS_HIGHER_POINT_AGREEMENT_THAN_R2_CORE"
                    )
            else:
                diagnosis = (
                    "R2_ALIGNMENT_VALID_BUT_CORE_COMPARISON_UNAVAILABLE"
                )

    print(
        f"Total mapped annotations : {total_annotations}"
    )
    print(
        f"Diagnosis                 : {diagnosis}"
    )

    print()
    print(
        "IMPORTANT: This diagnosis is based on the corrected Part 67-style "
        "annotation extraction. It replaces the invalid zero-annotation "
        "agreement result from Part 68."
    )

    print()
    print(
        "Scientific limitation: RSNA coordinates are point annotations, "
        "not manual segmentation masks. These measurements assess "
        "pseudo-target consistency and boundary sensitivity, not clinical "
        "segmentation accuracy."
    )

    # ========================================================================
    # OUTPUTS
    # ========================================================================

    REPORT.mkdir(
        parents=True,
        exist_ok=True,
    )

    region_df.to_csv(
        REPORT / "part68_1_region_case_statistics.csv",
        index=False,
    )

    region_summary_df.to_csv(
        REPORT / "part68_1_region_statistics.csv",
        index=False,
    )

    point_df.to_csv(
        REPORT / "part68_1_annotation_agreement_by_region.csv",
        index=False,
    )

    agreement_df.to_csv(
        REPORT / "part68_1_annotation_agreement_summary.csv",
        index=False,
    )

    class_df.to_csv(
        REPORT / "part68_1_per_class_region_agreement.csv",
        index=False,
    )

    summary = {
        "part": "68.1",
        "status": "CORRECTED",
        "validation_subset": N,
        "full_shape": list(FULL),
        "crop_shape": list(CROP),
        "radii": list(RADII),
        "core_erosion_iterations": CORE_EROSION_ITERATIONS,
        "total_mapped_annotations": int(total_annotations),
        "r2_mean_foreground_voxels": r2_fg,
        "r2_core_mean_foreground_voxels": core_fg,
        "r2_halo_mean_foreground_voxels": halo_fg,
        "r2_core_fraction": core_fraction,
        "r2_halo_fraction": halo_fraction,
        "diagnosis": diagnosis,
        "training_performed": False,
        "optimizer_used": False,
        "backward_pass": False,
        "spider_used": False,
        "test_set_used": False,
        "part15_modified": False,
        "correction": (
            "Annotation extraction and point mapping reuse the exact "
            "Part 67 logic; previous Part 68 zero-annotation agreement "
            "results are discarded."
        ),
        "scientific_limitation": (
            "RSNA coordinates are point annotations, not manual "
            "segmentation masks."
        ),
    }

    with open(
        REPORT / "part68_1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    report_lines = [
        "PART 68.1 — CORRECTED PSEUDO-MASK RADIUS / BOUNDARY STABILITY",
        "",
        f"Validation subset: {N}",
        f"Total mapped annotations: {total_annotations}",
        f"Diagnosis: {diagnosis}",
        "",
        "Region statistics:",
    ]

    for _, r in region_summary_df.iterrows():
        report_lines.append(
            f"{r['region']}: "
            f"mean FG={r['mean_foreground_voxels']:.2f}, "
            f"fraction={r['mean_foreground_fraction']:.8f}, "
            f"components={r['mean_connected_components']:.2f}, "
            f"active-Z={r['mean_active_z_fraction']:.4f}"
        )

    report_lines.extend(
        [
            "",
            "R2 core/halo:",
            f"R2 core fraction: {core_fraction:.6f}",
            f"R2 halo fraction: {halo_fraction:.6f}",
            "",
            "Scientific limitation:",
            "RSNA coordinates are point annotations, not manual "
            "segmentation masks. This is a target-consistency and "
            "boundary-sensitivity diagnostic, not medical segmentation "
            "accuracy.",
        ]
    )

    (REPORT / "part68_1_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    banner("PART 68.1 COMPLETE")

    print(
        f"Region case CSV       : {REPORT / 'part68_1_region_case_statistics.csv'}"
    )
    print(
        f"Region summary CSV    : {REPORT / 'part68_1_region_statistics.csv'}"
    )
    print(
        f"Annotation detail CSV : {REPORT / 'part68_1_annotation_agreement_by_region.csv'}"
    )
    print(
        f"Agreement summary CSV : {REPORT / 'part68_1_annotation_agreement_summary.csv'}"
    )
    print(
        f"Per-class CSV         : {REPORT / 'part68_1_per_class_region_agreement.csv'}"
    )
    print(
        f"Summary JSON          : {REPORT / 'part68_1_summary.json'}"
    )
    print(
        f"Text report           : {REPORT / 'part68_1_report.txt'}"
    )


if __name__ == "__main__":
    main()
