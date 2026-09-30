"""
Part 4.6C — Quantitative 3D XAI Summary & Visualization

Purpose
-------
Analyze the already-completed Part 4.6B multi-point XAI results.

This module:
1. Loads Part 4.6B point_results.csv
2. Produces disease-wise statistics
3. Produces level-wise statistics
4. Produces overall statistics
5. Generates comparison plots
6. Generates a JSON summary
7. Generates a human-readable research report

IMPORTANT
---------
- No model is loaded.
- No training is performed.
- No checkpoint is modified.
- No geometry pipeline is modified.
- Part 4.6B results are treated as frozen experimental results.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
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
    / "rsna_part46c_xai_summary"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = INPUT_DIR / "point_results.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

DISEASE_ORDER = [
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
]

LEVEL_ORDER = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1",
]

METRIC_COLUMNS = [
    "centroid_distance",
    "max_attribution_distance",
    "r2_concentration",
    "r4_concentration",
    "r6_concentration",
    "r10_concentration",
]


# ============================================================
# 3. HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_column(df, candidates, required=True):
    """
    Find the first matching column from a list of possible names.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def clean_disease_name(value):
    """
    Normalize disease labels.
    """

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip()

    mapping = {
        "left_neural_foraminal_narrowing": "LFNN",
        "right_neural_foraminal_narrowing": "RFNN",
        "LFNN": "LFNN",
        "RFNN": "RFNN",
        "lfnn": "LFNN",
        "rfnn": "RFNN",
    }

    return mapping.get(value, value)


def clean_level(value):
    """
    Normalize spinal level labels.
    """

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip()

    replacements = {
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

    return replacements.get(value, value)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# 4. LOAD PART 4.6B RESULTS
# ============================================================

print_header("PART 4.6C — QUANTITATIVE XAI SUMMARY")

print(f"Project root : {PROJECT_ROOT}")
print(f"Input file   : {INPUT_CSV}")
print(f"Output dir   : {OUTPUT_DIR}")

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"\nPart 4.6B result file was not found:\n{INPUT_CSV}\n\n"
        "Run Part 4.6B successfully before running Part 4.6C."
    )

df = pd.read_csv(INPUT_CSV)

print(f"\nLoaded rows: {len(df)}")

print("\nAvailable columns:")
for column in df.columns:
    print(f"  - {column}")


# ============================================================
# 5. IDENTIFY IMPORTANT COLUMNS
# ============================================================

disease_col = find_column(
    df,
    [
        "disease",
        "condition",
        "disease_name",
        "class_name",
    ],
)

level_col = find_column(
    df,
    [
        "level",
        "spinal_level",
    ],
)

centroid_col = find_column(
    df,
    [
        "centroid_distance",
    ],
)

max_distance_col = find_column(
    df,
    [
        "max_distance",
        "max_attribution_distance",
    ],
)

r2_col = find_column(
    df,
    [
        "r2_concentration",
        "r2",
    ],
)

r4_col = find_column(
    df,
    [
        "r4_concentration",
        "r4",
    ],
)

r6_col = find_column(
    df,
    [
        "r6_concentration",
        "r6",
    ],
)

r10_col = find_column(
    df,
    [
        "r10_concentration",
        "r10",
    ],
)


# ============================================================
# 6. NORMALIZE DATA
# ============================================================

df["disease_clean"] = df[disease_col].apply(clean_disease_name)
df["level_clean"] = df[level_col].apply(clean_level)

df["centroid_distance"] = df[centroid_col].apply(safe_float)
df["max_attribution_distance"] = df[max_distance_col].apply(safe_float)
df["r2_concentration"] = df[r2_col].apply(safe_float)
df["r4_concentration"] = df[r4_col].apply(safe_float)
df["r6_concentration"] = df[r6_col].apply(safe_float)
df["r10_concentration"] = df[r10_col].apply(safe_float)


# ============================================================
# 7. BASIC DATA QUALITY CHECK
# ============================================================

print_header("DATA QUALITY CHECK")

print(f"Total XAI points: {len(df)}")

print("\nDisease distribution:")
print(df["disease_clean"].value_counts())

print("\nSpinal-level distribution:")
print(df["level_clean"].value_counts().sort_index())

print("\nMissing values:")
print(
    df[
        [
            "disease_clean",
            "level_clean",
            *METRIC_COLUMNS,
        ]
    ].isna().sum()
)


# ============================================================
# 8. OVERALL SUMMARY
# ============================================================

overall_rows = []

for metric in METRIC_COLUMNS:
    values = df[metric].dropna()

    if len(values) == 0:
        continue

    overall_rows.append(
        {
            "metric": metric,
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )

overall_summary = pd.DataFrame(overall_rows)


# ============================================================
# 9. DISEASE-WISE SUMMARY
# ============================================================

print_header("DISEASE-WISE SUMMARY")

disease_summary = (
    df.groupby("disease_clean")[METRIC_COLUMNS]
    .agg(["count", "mean", "std", "median", "min", "max"])
)

print(disease_summary)


# ============================================================
# 10. FLAT DISEASE-WISE TABLE
# ============================================================

disease_flat_rows = []

for disease in DISEASE_ORDER:

    subset = df[df["disease_clean"] == disease]

    if subset.empty:
        continue

    row = {
        "disease": disease,
        "n_points": int(len(subset)),
    }

    for metric in METRIC_COLUMNS:

        values = subset[metric].dropna()

        if len(values) == 0:
            row[f"{metric}_mean"] = np.nan
            row[f"{metric}_std"] = np.nan
            row[f"{metric}_median"] = np.nan
        else:
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_median"] = float(values.median())

    disease_flat_rows.append(row)

disease_flat = pd.DataFrame(disease_flat_rows)

print("\nFlat disease summary:")
print(disease_flat.to_string(index=False))


# ============================================================
# 11. LEVEL-WISE SUMMARY
# ============================================================

print_header("LEVEL-WISE SUMMARY")

level_flat_rows = []

for level in LEVEL_ORDER:

    subset = df[df["level_clean"] == level]

    if subset.empty:
        continue

    row = {
        "level": level,
        "n_points": int(len(subset)),
    }

    for metric in METRIC_COLUMNS:

        values = subset[metric].dropna()

        if len(values) == 0:
            row[f"{metric}_mean"] = np.nan
            row[f"{metric}_std"] = np.nan
            row[f"{metric}_median"] = np.nan
        else:
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_median"] = float(values.median())

    level_flat_rows.append(row)

level_flat = pd.DataFrame(level_flat_rows)

print("\nFlat level summary:")
print(level_flat.to_string(index=False))


# ============================================================
# 12. DISEASE × LEVEL SUMMARY
# ============================================================

disease_level_summary = (
    df.groupby(
        [
            "disease_clean",
            "level_clean",
        ]
    )[METRIC_COLUMNS]
    .mean()
    .reset_index()
)

print_header("DISEASE × LEVEL SUMMARY")

print(disease_level_summary.to_string(index=False))


# ============================================================
# 13. SAVE CSV FILES
# ============================================================

overall_summary.to_csv(
    OUTPUT_DIR / "part46c_overall_summary.csv",
    index=False,
)

disease_flat.to_csv(
    OUTPUT_DIR / "part46c_disease_summary.csv",
    index=False,
)

level_flat.to_csv(
    OUTPUT_DIR / "part46c_level_summary.csv",
    index=False,
)

disease_level_summary.to_csv(
    OUTPUT_DIR / "part46c_disease_level_summary.csv",
    index=False,
)

# Main requested summary file
disease_flat.to_csv(
    OUTPUT_DIR / "part46c_summary.csv",
    index=False,
)


# ============================================================
# 14. DISEASE COMPARISON PLOT
# ============================================================

print_header("GENERATING DISEASE COMPARISON PLOT")

plot_metrics = [
    ("centroid_distance", "Centroid Distance"),
    ("max_attribution_distance", "Max Attribution Distance"),
    ("r10_concentration", "R10 Attribution Concentration"),
]

figures_created = []

for metric, title in plot_metrics:

    if metric not in df.columns:
        continue

    grouped = []

    for disease in DISEASE_ORDER:
        values = df.loc[
            df["disease_clean"] == disease,
            metric,
        ].dropna()

        grouped.append(values.to_numpy())

    valid_groups = [
        values
        for values in grouped
        if len(values) > 0
    ]

    valid_labels = [
        DISEASE_ORDER[i]
        for i, values in enumerate(grouped)
        if len(values) > 0
    ]

    if not valid_groups:
        continue

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        valid_groups,
        labels=valid_labels,
    )

    plt.title(
        f"Part 4.6C — Disease-wise {title}"
    )

    plt.ylabel(title)
    plt.xlabel("Disease")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    filename = (
        OUTPUT_DIR
        / f"part46c_{metric}_comparison.png"
    )

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    figures_created.append(str(filename))

# Combined requested comparison image
fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 5),
)

for ax, (metric, title) in zip(
    axes,
    plot_metrics,
):

    grouped = []
    labels = []

    for disease in DISEASE_ORDER:

        values = df.loc[
            df["disease_clean"] == disease,
            metric,
        ].dropna()

        if len(values) > 0:
            grouped.append(values.to_numpy())
            labels.append(disease)

    if grouped:

        ax.boxplot(
            grouped,
            labels=labels,
        )

        ax.set_title(title)
        ax.set_xlabel("Disease")
        ax.set_ylabel(title)
        ax.grid(
            axis="y",
            alpha=0.25,
        )

fig.suptitle(
    "Part 4.6C — Quantitative XAI Disease Comparison",
    fontsize=14,
)

fig.tight_layout()

comparison_path = (
    OUTPUT_DIR
    / "part46c_comparison.png"
)

fig.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)

figures_created.append(
    str(comparison_path)
)


# ============================================================
# 15. LEVEL-WISE R10 PLOT
# ============================================================

print_header("GENERATING LEVEL-WISE PLOT")

level_r10 = (
    df.groupby("level_clean")["r10_concentration"]
    .mean()
)

level_r10 = level_r10.reindex(
    LEVEL_ORDER
).dropna()

plt.figure(figsize=(10, 6))

plt.plot(
    level_r10.index,
    level_r10.values,
    marker="o",
)

plt.title(
    "Part 4.6C — Mean R10 Attribution by Spinal Level"
)

plt.xlabel("Spinal Level")
plt.ylabel("Mean R10 Attribution Concentration")

plt.grid(
    axis="y",
    alpha=0.25,
)

plt.tight_layout()

level_plot_path = (
    OUTPUT_DIR
    / "part46c_levelwise.png"
)

plt.savefig(
    level_plot_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

figures_created.append(
    str(level_plot_path)
)


# ============================================================
# 16. DISEASE × LEVEL HEATMAP
# ============================================================

print_header("GENERATING DISEASE × LEVEL HEATMAP")

heatmap_data = (
    df.pivot_table(
        index="disease_clean",
        columns="level_clean",
        values="r10_concentration",
        aggfunc="mean",
    )
    .reindex(index=DISEASE_ORDER)
    .reindex(columns=LEVEL_ORDER)
)

if not heatmap_data.empty:

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(heatmap_data.columns))
    )

    ax.set_xticklabels(
        heatmap_data.columns
    )

    ax.set_yticks(
        range(len(heatmap_data.index))
    )

    ax.set_yticklabels(
        heatmap_data.index
    )

    ax.set_xlabel("Spinal Level")
    ax.set_ylabel("Disease")

    ax.set_title(
        "Mean R10 Attribution Concentration"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="R10 concentration",
    )

    for i in range(
        heatmap_data.shape[0]
    ):
        for j in range(
            heatmap_data.shape[1]
        ):

            value = heatmap_data.iloc[i, j]

            if not pd.isna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()

    heatmap_path = (
        OUTPUT_DIR
        / "part46c_disease_level_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    figures_created.append(
        str(heatmap_path)
    )


# ============================================================
# 17. JSON SUMMARY
# ============================================================

def dataframe_to_records(dataframe):
    """
    Convert NaN to None for valid JSON.
    """

    cleaned = dataframe.replace(
        {np.nan: None}
    )

    return cleaned.to_dict(
        orient="records"
    )


json_summary = {
    "experiment": "Part 4.6C — Quantitative 3D XAI Summary",
    "input_experiment": "Part 4.6B — Multi-Point / Multi-Level Point-Targeted 3D XAI",
    "input_file": str(INPUT_CSV),
    "total_points": int(len(df)),
    "disease_counts": {
        str(k): int(v)
        for k, v in df["disease_clean"]
        .value_counts()
        .items()
    },
    "level_counts": {
        str(k): int(v)
        for k, v in df["level_clean"]
        .value_counts()
        .items()
    },
    "overall": dataframe_to_records(
        overall_summary
    ),
    "disease_wise": dataframe_to_records(
        disease_flat
    ),
    "level_wise": dataframe_to_records(
        level_flat
    ),
    "disease_level": dataframe_to_records(
        disease_level_summary
    ),
    "generated_figures": figures_created,
    "interpretation": {
        "purpose": (
            "Quantify spatial concentration and "
            "distance characteristics of point-targeted "
            "3D XAI attributions."
        ),
        "clinical_localization_claim": (
            "No clinical localization claim is made "
            "from these measurements alone."
        ),
    },
}

JSON_PATH = (
    OUTPUT_DIR
    / "part46c_summary.json"
)

with open(
    JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_summary,
        f,
        indent=2,
    )


# ============================================================
# 18. RESEARCH REPORT
# ============================================================

report_lines = []

report_lines.append(
    "PART 4.6C — QUANTITATIVE 3D XAI SUMMARY"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    ""
)

report_lines.append(
    "Purpose:"
)

report_lines.append(
    "Part 4.6C summarizes the frozen results produced "
    "by Part 4.6B Multi-Point / Multi-Level Point-Targeted "
    "3D XAI validation."
)

report_lines.append(
    ""
)

report_lines.append(
    f"Total evaluated points: {len(df)}"
)

report_lines.append(
    ""
)

report_lines.append(
    "Disease distribution:"
)

for disease, count in (
    df["disease_clean"]
    .value_counts()
    .items()
):

    report_lines.append(
        f"  {disease}: {count}"
    )

report_lines.append(
    ""
)

report_lines.append(
    "OVERALL METRICS"
)

report_lines.append(
    "-" * 70
)

for _, row in overall_summary.iterrows():

    report_lines.append(
        f"{row['metric']}: "
        f"mean={row['mean']:.6f}, "
        f"std={row['std']:.6f}, "
        f"median={row['median']:.6f}, "
        f"min={row['min']:.6f}, "
        f"max={row['max']:.6f}"
    )

report_lines.append(
    ""
)

report_lines.append(
    "DISEASE-WISE METRICS"
)

report_lines.append(
    "-" * 70
)

for _, row in disease_flat.iterrows():

    report_lines.append(
        f"\n{row['disease']} "
        f"(n={int(row['n_points'])})"
    )

    for metric in METRIC_COLUMNS:

        mean_value = row[
            f"{metric}_mean"
        ]

        std_value = row[
            f"{metric}_std"
        ]

        if pd.isna(mean_value):
            continue

        report_lines.append(
            f"  {metric}: "
            f"{mean_value:.6f} "
            f"+/- {std_value:.6f}"
        )

report_lines.append(
    ""
)

report_lines.append(
    "LEVEL-WISE METRICS"
)

report_lines.append(
    "-" * 70
)

for _, row in level_flat.iterrows():

    report_lines.append(
        f"\n{row['level']} "
        f"(n={int(row['n_points'])})"
    )

    for metric in METRIC_COLUMNS:

        mean_value = row[
            f"{metric}_mean"
        ]

        if pd.isna(mean_value):
            continue

        report_lines.append(
            f"  {metric}: "
            f"{mean_value:.6f}"
        )

report_lines.append(
    ""
)

report_lines.append(
    "INTERPRETATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "The Part 4.6C analysis quantifies the spatial "
    "distribution of point-targeted 3D XAI attribution."
)

report_lines.append(
    "Centroid distance measures the distance between "
    "the annotated target point and the attribution centroid."
)

report_lines.append(
    "Maximum attribution distance measures the distance "
    "between the target point and the strongest attribution voxel."
)

report_lines.append(
    "R2, R4, R6 and R10 quantify the proportion of attribution "
    "concentrated within increasingly larger neighborhoods "
    "around the target point."
)

report_lines.append(
    ""
)

report_lines.append(
    "IMPORTANT LIMITATION"
)

report_lines.append(
    "-" * 70
)

report_lines.append(
    "These measurements do not establish clinical validity "
    "or clinically meaningful anatomical localization."
)

report_lines.append(
    "The results should be interpreted as quantitative "
    "model-attribution diagnostics for the evaluated cases."
)

report_lines.append(
    ""
)

report_lines.append(
    "Experiment status:"
)

report_lines.append(
    "No training was performed."
)

report_lines.append(
    "No checkpoint was modified."
)

report_lines.append(
    "No physical-space geometry was modified."
)

REPORT_PATH = (
    OUTPUT_DIR
    / "part46c_report.txt"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# 19. FINAL OUTPUT SUMMARY
# ============================================================

print_header("PART 4.6C COMPLETE")

print(
    "\nGenerated files:"
)

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    if path.is_file():

        print(
            f"  {path.name}"
        )

print(
    "\nPart 4.6C completed successfully."
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