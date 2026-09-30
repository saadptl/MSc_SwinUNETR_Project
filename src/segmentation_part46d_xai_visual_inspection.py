"""
Part 4.6D — Visual / Anatomical Inspection of Point-Targeted 3D XAI

Purpose
-------
Inspect the frozen Part 4.6B multi-point XAI attribution maps.

This module:
1. Loads Part 4.6B point_results.csv
2. Loads each saved XAI .npy volume
3. Creates clean axial/coronal/sagittal inspection panels
4. Marks the annotated target point
5. Compares LFNN and RFNN visually
6. Produces per-point inspection images
7. Produces disease-level overview images
8. Generates a structured inspection report
9. Generates a JSON summary

IMPORTANT
---------
- No model is loaded.
- No training is performed.
- No checkpoint is modified.
- No geometry is modified.
- Part 4.6B results are treated as frozen.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part46b_multi_point_3d_xai"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part46d_xai_visual_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INPUT_CSV = INPUT_DIR / "point_results.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

DISEASE_ORDER = [
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
]

DISEASE_SHORT = {
    "Left Neural Foraminal Narrowing": "LFNN",
    "Right Neural Foraminal Narrowing": "RFNN",
}

LEVEL_ORDER = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]


# ============================================================
# 3. HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def clean_level(value):
    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip()

    mapping = {
        "L1-L2": "L1/L2",
        "L2-L3": "L2/L3",
        "L3-L4": "L3/L4",
        "L4-L5": "L4/L5",
        "L5-S1": "L5/S1",
        "L1/L2": "L1/L2",
        "L2/L3": "L2/L3",
        "L3/L4": "L3/L4",
        "L4/L5": "L4/L5",
        "L5/S1": "L5/S1",
    }

    return mapping.get(value, value)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def normalize_xai(xai):
    """
    Normalize an XAI volume to [0, 1].

    The saved Part 4.6B XAI maps are expected to already be
    normalized, but this makes the inspection robust.
    """

    xai = np.asarray(
        xai,
        dtype=np.float32,
    )

    xai = np.nan_to_num(
        xai,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(xai.min())
    maximum = float(xai.max())

    if maximum <= minimum:
        return np.zeros_like(xai)

    xai = (xai - minimum) / (
        maximum - minimum
    )

    return xai


def load_xai(path):
    """
    Load and validate one XAI volume.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"XAI file not found: {path}"
        )

    xai = np.load(path)

    if xai.ndim != 3:
        raise ValueError(
            f"Expected 3D XAI array, got shape {xai.shape}"
        )

    return normalize_xai(xai)


def safe_int_coordinate(value, maximum):
    """
    Convert a coordinate to a valid voxel index.
    """

    value = int(
        round(
            float(value)
        )
    )

    return int(
        np.clip(
            value,
            0,
            maximum - 1,
        )
    )


def percentile_threshold(xai, percentile=95.0):
    """
    Compute a high-attribution threshold.
    """

    finite_values = xai[
        np.isfinite(xai)
    ]

    if finite_values.size == 0:
        return 0.0

    return float(
        np.percentile(
            finite_values,
            percentile,
        )
    )


def calculate_visual_metrics(
    xai,
    z,
    y,
    x,
):
    """
    Calculate simple visual/anatomical diagnostics.

    These are descriptive diagnostics only.
    They do NOT establish clinical localization.
    """

    target = np.array(
        [
            float(z),
            float(y),
            float(x),
        ],
        dtype=np.float32,
    )

    grid_z, grid_y, grid_x = np.indices(
        xai.shape
    )

    distances = np.sqrt(
        (grid_z - target[0]) ** 2
        + (grid_y - target[1]) ** 2
        + (grid_x - target[2]) ** 2
    )

    total_attribution = float(
        xai.sum()
    )

    if total_attribution > 0:

        weighted_z = float(
            (xai * grid_z).sum()
            / total_attribution
        )

        weighted_y = float(
            (xai * grid_y).sum()
            / total_attribution
        )

        weighted_x = float(
            (xai * grid_x).sum()
            / total_attribution
        )

        centroid_distance = float(
            np.sqrt(
                (weighted_z - target[0]) ** 2
                + (weighted_y - target[1]) ** 2
                + (weighted_x - target[2]) ** 2
            )
        )

    else:

        weighted_z = np.nan
        weighted_y = np.nan
        weighted_x = np.nan
        centroid_distance = np.nan

    max_index = np.unravel_index(
        np.argmax(xai),
        xai.shape,
    )

    max_distance = float(
        distances[max_index]
    )

    threshold = percentile_threshold(
        xai,
        percentile=95.0,
    )

    high_mask = xai >= threshold

    high_voxel_count = int(
        high_mask.sum()
    )

    if high_voxel_count > 0:

        high_mean_distance = float(
            distances[high_mask].mean()
        )

    else:

        high_mean_distance = np.nan

    return {
        "xai_min": float(xai.min()),
        "xai_max": float(xai.max()),
        "xai_mean": float(xai.mean()),
        "xai_sum": total_attribution,
        "p95_threshold": threshold,
        "high_attribution_voxels": high_voxel_count,
        "high_attribution_mean_distance": high_mean_distance,
        "centroid_z": weighted_z,
        "centroid_y": weighted_y,
        "centroid_x": weighted_x,
        "centroid_distance": centroid_distance,
        "max_z": int(max_index[0]),
        "max_y": int(max_index[1]),
        "max_x": int(max_index[2]),
        "max_distance": max_distance,
    }


# ============================================================
# 4. LOAD PART 4.6B RESULTS
# ============================================================

print_header(
    "PART 4.6D — VISUAL / ANATOMICAL XAI INSPECTION"
)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Input CSV    : {INPUT_CSV}"
)

print(
    f"Output dir   : {OUTPUT_DIR}"
)

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"\nPart 4.6B CSV was not found:\n"
        f"{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"\nLoaded Part 4.6B rows: {len(df)}"
)


# ============================================================
# 5. NORMALIZE DATA
# ============================================================

df["level_clean"] = (
    df["level"]
    .apply(clean_level)
)

df["disease"] = (
    df["disease"]
    .astype(str)
    .str.strip()
)

df["point_z"] = df["point_z"].apply(
    safe_float
)

df["point_y"] = df["point_y"].apply(
    safe_float
)

df["point_x"] = df["point_x"].apply(
    safe_float
)


# ============================================================
# 6. CHECK DATASET
# ============================================================

print_header(
    "PART 4.6B INPUT CHECK"
)

print(
    f"Total points: {len(df)}"
)

print(
    "\nDisease counts:"
)

print(
    df["disease"].value_counts()
)

print(
    "\nLevel counts:"
)

print(
    df["level_clean"].value_counts()
    .sort_index()
)

required_columns = [
    "point_index",
    "study_id",
    "series_id",
    "disease",
    "level_clean",
    "point_z",
    "point_y",
    "point_x",
    "xai_file",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise KeyError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# 7. OUTPUT SUBDIRECTORIES
# ============================================================

POINT_OUTPUT_DIR = (
    OUTPUT_DIR / "point_inspections"
)

DISEASE_OUTPUT_DIR = (
    OUTPUT_DIR / "disease_overviews"
)

POINT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DISEASE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 8. PROCESS EACH XAI POINT
# ============================================================

print_header(
    "PROCESSING XAI POINTS"
)

visual_records = []

for row_number, row in df.iterrows():

    disease = row["disease"]
    short_disease = DISEASE_SHORT.get(
        disease,
        disease,
    )

    level = row["level_clean"]

    point_index = int(
        row["point_index"]
    )

    study_id = str(
        row["study_id"]
    )

    series_id = str(
        row["series_id"]
    )

    z = safe_int_coordinate(
        row["point_z"],
        64,
    )

    y = safe_int_coordinate(
        row["point_y"],
        96,
    )

    x = safe_int_coordinate(
        row["point_x"],
        96,
    )

    xai_path = Path(
        str(row["xai_file"])
    )

    if not xai_path.is_absolute():

        xai_path = (
            PROJECT_ROOT
            / xai_path
        )

    # Handle paths that may already contain
    # the project output directory.
    if not xai_path.exists():

        alternative = (
            INPUT_DIR
            / xai_path.name
        )

        if alternative.exists():
            xai_path = alternative

    print(
        f"\n[{row_number + 1}/{len(df)}] "
        f"{short_disease} | {level}"
    )

    print(
        f"  Study  : {study_id}"
    )

    print(
        f"  Series : {series_id}"
    )

    print(
        f"  Point  : "
        f"(z={z}, y={y}, x={x})"
    )

    print(
        f"  XAI    : {xai_path}"
    )

    xai = load_xai(
        xai_path
    )

    depth, height, width = (
        xai.shape
    )

    z = safe_int_coordinate(
        z,
        depth,
    )

    y = safe_int_coordinate(
        y,
        height,
    )

    x = safe_int_coordinate(
        x,
        width,
    )

    metrics = calculate_visual_metrics(
        xai,
        z,
        y,
        x,
    )

    print(
        f"  XAI shape : {xai.shape}"
    )

    print(
        f"  XAI mean  : "
        f"{metrics['xai_mean']:.6f}"
    )

    print(
        f"  Centroid distance : "
        f"{metrics['centroid_distance']:.4f}"
    )

    print(
        f"  Max distance : "
        f"{metrics['max_distance']:.4f}"
    )

    print(
        f"  P95 threshold : "
        f"{metrics['p95_threshold']:.6f}"
    )

    # --------------------------------------------------------
    # SLICES
    # --------------------------------------------------------

    axial = xai[z, :, :]

    coronal = xai[:, y, :]

    sagittal = xai[:, :, x]

    # --------------------------------------------------------
    # CREATE 3-PANEL FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # Axial
    axes[0].imshow(
        axial,
        origin="lower",
        aspect="auto",
    )

    axes[0].scatter(
        [x],
        [y],
        marker="x",
        s=80,
    )

    axes[0].set_title(
        f"Axial | z={z}"
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Coronal
    axes[1].imshow(
        coronal,
        origin="lower",
        aspect="auto",
    )

    axes[1].scatter(
        [x],
        [z],
        marker="x",
        s=80,
    )

    axes[1].set_title(
        f"Coronal | y={y}"
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    # Sagittal
    axes[2].imshow(
        sagittal,
        origin="lower",
        aspect="auto",
    )

    axes[2].scatter(
        [y],
        [z],
        marker="x",
        s=80,
    )

    axes[2].set_title(
        f"Sagittal | x={x}"
    )

    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")

    fig.suptitle(
        f"{short_disease} — {level}\n"
        f"Study {study_id} | Series {series_id}\n"
        f"Target point = ({z}, {y}, {x})",
        fontsize=13,
    )

    fig.tight_layout()

    filename = (
        f"{short_disease}_"
        f"{level.replace('/', '-')}_"
        f"point_{point_index}.png"
    )

    output_path = (
        POINT_OUTPUT_DIR
        / filename
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    visual_record = {
        "point_index": point_index,
        "study_id": study_id,
        "series_id": series_id,
        "disease": disease,
        "disease_short": short_disease,
        "level": level,
        "point_z": z,
        "point_y": y,
        "point_x": x,
        "xai_shape": list(xai.shape),
        "visualization_file": str(
            output_path
        ),
        **metrics,
    }

    visual_records.append(
        visual_record
    )


# ============================================================
# 9. SAVE POINT-LEVEL VISUAL METRICS
# ============================================================

visual_df = pd.DataFrame(
    visual_records
)

visual_csv = (
    OUTPUT_DIR
    / "part46d_visual_metrics.csv"
)

visual_df.to_csv(
    visual_csv,
    index=False,
)


# ============================================================
# 10. DISEASE OVERVIEW FIGURES
# ============================================================

print_header(
    "CREATING DISEASE OVERVIEW FIGURES"
)

for disease in DISEASE_ORDER:

    disease_rows = visual_df[
        visual_df["disease"] == disease
    ]

    if disease_rows.empty:
        continue

    short_disease = DISEASE_SHORT[
        disease
    ]

    fig, axes = plt.subplots(
        1,
        len(disease_rows),
        figsize=(
            4 * len(disease_rows),
            5,
        ),
    )

    if len(disease_rows) == 1:
        axes = [axes]

    for ax, (_, item) in zip(
        axes,
        disease_rows.iterrows(),
    ):

        xai_path = Path(
            item["visualization_file"]
        )

        # The point visualization is already
        # generated, so this overview uses a
        # compact textual panel.
        ax.axis("off")

        text = (
            f"{short_disease}\n"
            f"{item['level']}\n\n"
            f"Target:\n"
            f"({item['point_z']}, "
            f"{item['point_y']}, "
            f"{item['point_x']})\n\n"
            f"Centroid distance:\n"
            f"{item['centroid_distance']:.2f}\n\n"
            f"Max distance:\n"
            f"{item['max_distance']:.2f}\n\n"
            f"R10:\n"
            f"{item.get('r10_concentration', np.nan):.6f}"
        )

        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.suptitle(
        f"{short_disease} — "
        f"Point-Targeted XAI Inspection Summary",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        DISEASE_OUTPUT_DIR
        / f"{short_disease}_overview.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 11. DISEASE COMPARISON PLOT
# ============================================================

print_header(
    "CREATING DISEASE COMPARISON"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

disease_labels = []
centroid_values = []
r10_values = []

for disease in DISEASE_ORDER:

    subset = visual_df[
        visual_df["disease"] == disease
    ]

    if subset.empty:
        continue

    disease_labels.append(
        DISEASE_SHORT[disease]
    )

    centroid_values.append(
        subset[
            "centroid_distance"
        ].mean()
    )

    # R10 may not be in visual_records,
    # so obtain it from original df.
    original_subset = df[
        df["disease"] == disease
    ]

    r10_values.append(
        original_subset[
            "r10_concentration"
        ].mean()
    )

axes[0].bar(
    disease_labels,
    centroid_values,
)

axes[0].set_title(
    "Mean XAI Centroid Distance"
)

axes[0].set_xlabel(
    "Disease"
)

axes[0].set_ylabel(
    "Distance (voxels)"
)

axes[0].grid(
    axis="y",
    alpha=0.25,
)

axes[1].bar(
    disease_labels,
    r10_values,
)

axes[1].set_title(
    "Mean R10 Attribution Concentration"
)

axes[1].set_xlabel(
    "Disease"
)

axes[1].set_ylabel(
    "R10 concentration"
)

axes[1].grid(
    axis="y",
    alpha=0.25,
)

fig.suptitle(
    "Part 4.6D — LFNN vs RFNN XAI Inspection",
    fontsize=14,
)

fig.tight_layout()

comparison_path = (
    OUTPUT_DIR
    / "part46d_disease_comparison.png"
)

fig.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 12. LEVEL-WISE CENTROID DISTANCE
# ============================================================

print_header(
    "CREATING LEVEL-WISE VISUAL SUMMARY"
)

level_summary = (
    visual_df.groupby(
        "level"
    )[
        "centroid_distance"
    ]
    .mean()
    .reindex(
        LEVEL_ORDER
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    level_summary.index,
    level_summary.values,
    marker="o",
)

plt.title(
    "Part 4.6D — Mean XAI Centroid Distance by Level"
)

plt.xlabel(
    "Spinal Level"
)

plt.ylabel(
    "Mean centroid distance (voxels)"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

level_path = (
    OUTPUT_DIR
    / "part46d_level_centroid_distance.png"
)

plt.savefig(
    level_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 13. VISUAL INSPECTION REPORT
# ============================================================

print_header(
    "GENERATING PART 4.6D REPORT"
)

report_lines = []

report_lines.append(
    "PART 4.6D — VISUAL / ANATOMICAL XAI INSPECTION"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "Input experiment: "
    "Part 4.6B Multi-Point / Multi-Level "
    "Point-Targeted 3D XAI"
)

report_lines.append(
    f"Total inspected points: {len(visual_df)}"
)

report_lines.append("")

report_lines.append(
    "Disease distribution:"
)

for disease, count in (
    visual_df[
        "disease"
    ].value_counts().items()
):

    report_lines.append(
        f"  {DISEASE_SHORT.get(disease, disease)}: "
        f"{count}"
    )

report_lines.append("")

report_lines.append(
    "LEVEL DISTRIBUTION"
)

report_lines.append(
    "-" * 70
)

for level in LEVEL_ORDER:

    count = int(
        (
            visual_df["level"]
            == level
        ).sum()
    )

    report_lines.append(
        f"{level}: {count}"
    )

report_lines.append("")

report_lines.append(
    "VISUAL / SPATIAL DIAGNOSTICS"
)

report_lines.append(
    "-" * 70
)

for disease in DISEASE_ORDER:

    subset = visual_df[
        visual_df["disease"] == disease
    ]

    if subset.empty:
        continue

    short_name = DISEASE_SHORT[
        disease
    ]

    mean_centroid = float(
        subset[
            "centroid_distance"
        ].mean()
    )

    mean_max_distance = float(
        subset[
            "max_distance"
        ].mean()
    )

    report_lines.append(
        f"\n{short_name}:"
    )

    report_lines.append(
        f"  Mean centroid distance: "
        f"{mean_centroid:.4f} voxels"
    )

    report_lines.append(
        f"  Mean maximum-attribution "
        f"distance: "
        f"{mean_max_distance:.4f} voxels"
    )

report_lines.append("")

report_lines.append(
    "POINT-BY-POINT INSPECTION"
)

report_lines.append(
    "-" * 70
)

for _, row in visual_df.iterrows():

    report_lines.append(
        f"{row['disease_short']} | "
        f"{row['level']} | "
        f"point=({row['point_z']}, "
        f"{row['point_y']}, "
        f"{row['point_x']}) | "
        f"centroid_distance="
        f"{row['centroid_distance']:.4f} | "
        f"max_distance="
        f"{row['max_distance']:.4f}"
    )

report_lines.append("")

report_lines.append(
    "INTERPRETATION GUIDANCE"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "The generated visualizations are intended "
    "to inspect whether model attribution is "
    "spatially associated with the annotated "
    "target point."
)

report_lines.append(
    "The target marker identifies the annotated "
    "canonical model-grid point."
)

report_lines.append(
    "Diffuse attribution should be interpreted "
    "as distributed model attribution rather "
    "than automatically as anatomical error."
)

report_lines.append(
    "Visual proximity alone does not establish "
    "clinical validity."
)

report_lines.append(
    "No clinical localization claim is made "
    "from Part 4.6D."
)

report_lines.append("")

report_lines.append(
    "EXPERIMENT INTEGRITY"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "No model training was performed."
)

report_lines.append(
    "No checkpoint was modified."
)

report_lines.append(
    "No geometry pipeline was modified."
)

report_lines.append(
    "Part 4.6B XAI outputs were treated as frozen."
)

report_path = (
    OUTPUT_DIR
    / "part46d_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# 14. JSON SUMMARY
# ============================================================

json_data = {
    "experiment": (
        "Part 4.6D — Visual / Anatomical "
        "XAI Inspection"
    ),
    "input_experiment": (
        "Part 4.6B — Multi-Point / "
        "Multi-Level Point-Targeted 3D XAI"
    ),
    "total_points": int(
        len(visual_df)
    ),
    "disease_counts": {
        str(k): int(v)
        for k, v in visual_df[
            "disease"
        ].value_counts().items()
    },
    "level_counts": {
        str(k): int(v)
        for k, v in visual_df[
            "level"
        ].value_counts().items()
    },
    "point_metrics": visual_df
        .replace(
            {np.nan: None}
        )
        .to_dict(
            orient="records"
        ),
    "output_files": [
        str(
            p
        )
        for p in OUTPUT_DIR.rglob("*")
        if p.is_file()
    ],
    "limitations": [
        "Visual inspection is descriptive.",
        "The evaluated set contains 10 points.",
        "Clinical localization validity is not established.",
        "No model retraining was performed.",
    ],
}

json_path = (
    OUTPUT_DIR
    / "part46d_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_data,
        f,
        indent=2,
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print_header(
    "PART 4.6D COMPLETE"
)

print(
    "\nGenerated output directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated files:"
)

for path in sorted(
    OUTPUT_DIR.rglob("*")
):

    if path.is_file():

        print(
            f"  {path.relative_to(OUTPUT_DIR)}"
        )

print(
    "\nNo model training was performed."
)

print(
    "No checkpoint was modified."
)

print(
    "No geometry pipeline was modified."
)

print(
    "\nPart 4.6D completed successfully."
)