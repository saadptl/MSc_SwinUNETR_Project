from pathlib import Path
import json
import re
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# PART 86
# RSNA DISEASE CLASSIFICATION DATASET AUDIT
# CORRECTED VERSION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_CSV = RSNA_ROOT / "train.csv"

COORD_CSV = RSNA_ROOT / "train_label_coordinates.csv"

SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"


OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part86_dataset_audit"
)

REPORT_DIR = ROOT / "reports"

LABEL_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part86_label_distribution.csv"
)

MISSING_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part86_missing_values.csv"
)

STUDY_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part86_study_summary.csv"
)

REPORT_TXT = (
    REPORT_DIR
    / "part86_rsna_classification_dataset_audit_report.txt"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part86_rsna_classification_dataset_audit_summary.json"
)


# ============================================================
# RSNA TARGET STRUCTURE
# ============================================================

LEVELS = [
    "l1_l2",
    "l2_l3",
    "l3_l4",
    "l4_l5",
    "l5_s1",
]

CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]


EXPECTED_TARGETS = [
    f"{condition}_{level}"
    for condition in CONDITIONS
    for level in LEVELS
]


CLASS_NAMES = {
    "spinal_canal_stenosis": "Spinal Canal Stenosis",
    "left_neural_foraminal_narrowing":
        "Left Neural Foraminal Narrowing",
    "right_neural_foraminal_narrowing":
        "Right Neural Foraminal Narrowing",
    "left_subarticular_stenosis":
        "Left Subarticular Stenosis",
    "right_subarticular_stenosis":
        "Right Subarticular Stenosis",
}


SEVERITY_LABELS = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]


SEVERITY_ALIASES = {
    "normal/mild": "Normal/Mild",
    "normal": "Normal/Mild",
    "mild": "Normal/Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


# ============================================================
# UTILITIES
# ============================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def normalize_column_name(name):

    name = str(name).strip()

    name = name.replace(
        "\ufeff",
        ""
    )

    name = name.replace(
        " ",
        "_"
    )

    name = name.replace(
        "-",
        "_"
    )

    name = name.replace(
        "/",
        "_"
    )

    name = re.sub(
        r"_+",
        "_",
        name
    )

    return name.lower()


def normalize_label(value):

    if pd.isna(value):
        return "MISSING"

    text = str(value).strip()

    if not text:
        return "MISSING"

    key = text.lower()

    return SEVERITY_ALIASES.get(
        key,
        text
    )


def safe_percentage(
    count,
    total
):

    if total == 0:
        return 0.0

    return (
        float(count)
        / float(total)
        * 100.0
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    banner("PART 86 PATH VALIDATION")

    required = {
        "RSNA root": RSNA_ROOT,
        "train.csv": TRAIN_CSV,
        "train_label_coordinates.csv": COORD_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
    }

    missing = []

    for name, path in required.items():

        exists = path.exists()

        print(
            f"{name:<38}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "\nRequired RSNA file(s) missing:\n"
            + "\n".join(missing)
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_rsna_data():

    banner("LOADING RSNA CLASSIFICATION DATA")

    print(
        f"train.csv:\n{TRAIN_CSV}"
    )

    df = pd.read_csv(
        TRAIN_CSV
    )

    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]

    print(
        f"\nRows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    return df


# ============================================================
# TARGET AUDIT
# ============================================================

def detect_targets(df):

    detected = [
        target
        for target in EXPECTED_TARGETS
        if target in df.columns
    ]

    missing = [
        target
        for target in EXPECTED_TARGETS
        if target not in df.columns
    ]

    return detected, missing


# ============================================================
# IDENTIFIER AUDIT
# ============================================================

def identifier_audit(df):

    result = {}

    if "study_id" in df.columns:

        result["study_id_column"] = True

        result["study_rows"] = int(
            len(df)
        )

        result["unique_studies"] = int(
            df["study_id"].nunique()
        )

        result["duplicate_study_rows"] = int(
            df["study_id"]
            .duplicated(
                keep=False
            )
            .sum()
        )

    else:

        result["study_id_column"] = False
        result["study_rows"] = int(len(df))
        result["unique_studies"] = None
        result["duplicate_study_rows"] = None

    return result


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def label_distribution(
    df,
    targets
):

    rows = []

    global_counts = Counter()

    for target in targets:

        normalized = df[target].map(
            normalize_label
        )

        counts = normalized.value_counts(
            dropna=False
        )

        total = len(normalized)

        for label, count in counts.items():

            count = int(count)

            rows.append(
                {
                    "target": target,
                    "label": str(label),
                    "count": count,
                    "percentage":
                        safe_percentage(
                            count,
                            total
                        ),
                }
            )

            global_counts[str(label)] += count

    return (
        pd.DataFrame(rows),
        global_counts
    )


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

def missing_audit(
    df,
    targets
):

    rows = []

    for target in targets:

        missing = int(
            df[target].isna().sum()
        )

        rows.append(
            {
                "target": target,
                "missing_count": missing,
                "missing_percentage":
                    safe_percentage(
                        missing,
                        len(df)
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CASE COMPLETENESS
# ============================================================

def completeness_audit(
    df,
    targets
):

    complete_mask = (
        df[targets]
        .notna()
        .all(axis=1)
    )

    complete = int(
        complete_mask.sum()
    )

    incomplete = int(
        (~complete_mask).sum()
    )

    total = len(df)

    return {
        "total_studies": total,
        "complete_studies": complete,
        "incomplete_studies": incomplete,
        "complete_percentage":
            safe_percentage(
                complete,
                total
            ),
        "incomplete_percentage":
            safe_percentage(
                incomplete,
                total
            ),
    }


# ============================================================
# CONDITION-LEVEL SUMMARY
# ============================================================

def condition_summary(
    label_df
):

    rows = []

    for condition in CONDITIONS:

        condition_targets = [
            target
            for target in EXPECTED_TARGETS
            if target.startswith(
                condition + "_"
            )
        ]

        subset = label_df[
            label_df["target"].isin(
                condition_targets
            )
        ]

        for label in [
            "Normal/Mild",
            "Moderate",
            "Severe",
            "MISSING",
        ]:

            count = int(
                subset.loc[
                    subset["label"] == label,
                    "count"
                ].sum()
            )

            rows.append(
                {
                    "condition": condition,
                    "condition_name":
                        CLASS_NAMES[condition],
                    "label": label,
                    "count": count,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    df,
    targets,
    missing_targets,
    ids,
    completeness,
    label_df,
    missing_df,
    condition_df
):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "PART 86 — RSNA DISEASE CLASSIFICATION DATASET AUDIT"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        f"Project root: {ROOT}"
    )

    lines.append(
        f"RSNA dataset: {RSNA_ROOT}"
    )

    lines.append(
        f"train.csv: {TRAIN_CSV}"
    )

    lines.append("")

    lines.append(
        "DATASET SIZE"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Studies/rows: {len(df)}"
    )

    lines.append(
        f"Columns: {len(df.columns)}"
    )

    lines.append("")

    lines.append(
        "CLASSIFICATION TARGET AUDIT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Expected targets: 25"
    )

    lines.append(
        f"Detected targets: {len(targets)}"
    )

    for target in targets:

        lines.append(
            f"  [FOUND] {target}"
        )

    for target in missing_targets:

        lines.append(
            f"  [MISSING] {target}"
        )

    lines.append("")

    lines.append(
        "IDENTIFIER AUDIT"
    )

    lines.append(
        "-" * 78
    )

    for key, value in ids.items():

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "CASE COMPLETENESS"
    )

    lines.append(
        "-" * 78
    )

    for key, value in completeness.items():

        if "percentage" in key:

            lines.append(
                f"{key}: {value:.2f}%"
            )

        else:

            lines.append(
                f"{key}: {value}"
            )

    lines.append("")

    lines.append(
        "CONDITION-LEVEL LABEL SUMMARY"
    )

    lines.append(
        "-" * 78
    )

    for condition in CONDITIONS:

        name = CLASS_NAMES[condition]

        lines.append(
            f"\n{name}"
        )

        subset = condition_df[
            condition_df["condition"] == condition
        ]

        for _, row in subset.iterrows():

            lines.append(
                f"  "
                f"{row['label']:12s}"
                f" : "
                f"{int(row['count']):5d}"
            )

    lines.append("")

    lines.append(
        "TARGET-LEVEL LABEL DISTRIBUTION"
    )

    lines.append(
        "-" * 78
    )

    for target in targets:

        lines.append(
            f"\n{target}"
        )

        subset = label_df[
            label_df["target"] == target
        ]

        for _, row in subset.iterrows():

            lines.append(
                f"  "
                f"{row['label']:12s}"
                f" : "
                f"{int(row['count']):5d}"
                f" "
                f"({float(row['percentage']):6.2f}%)"
            )

    lines.append("")

    lines.append(
        "FINAL STATUS"
    )

    lines.append(
        "-" * 78
    )

    if len(targets) == 25:

        lines.append(
            "PASS: All 25 RSNA classification targets "
            "were detected."
        )

    else:

        lines.append(
            "REVIEW REQUIRED: Not all 25 expected "
            "classification targets were detected."
        )

    if completeness["incomplete_studies"] > 0:

        lines.append(
            "INFO: Some studies contain missing labels."
        )

    else:

        lines.append(
            "PASS: All studies contain complete "
            "classification labels."
        )

    lines.append("")

    lines.append(
        "Part 86 performs dataset auditing only."
    )

    lines.append(
        "No neural-network training was performed."
    )

    lines.append(
        "No Part 84/85 segmentation checkpoint was modified."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 86 — RSNA DISEASE CLASSIFICATION DATASET AUDIT"
    )

    print(
        f"\nProject root:\n{ROOT}"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Load train.csv
    # --------------------------------------------------------

    df = load_rsna_data()

    # --------------------------------------------------------
    # Print all columns
    # --------------------------------------------------------

    print(
        "\ntrain.csv columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    # --------------------------------------------------------
    # Detect targets
    # --------------------------------------------------------

    targets, missing_targets = detect_targets(
        df
    )

    banner(
        "CLASSIFICATION TARGET AUDIT"
    )

    print(
        f"Expected targets : 25"
    )

    print(
        f"Detected targets : {len(targets)}"
    )

    for target in targets:

        print(
            f"  [FOUND] {target}"
        )

    if missing_targets:

        print(
            "\nMissing expected targets:"
        )

        for target in missing_targets:

            print(
                f"  [MISSING] {target}"
            )

    if len(targets) == 0:

        raise RuntimeError(
            "No RSNA classification targets were detected."
        )

    # --------------------------------------------------------
    # Identifier audit
    # --------------------------------------------------------

    ids = identifier_audit(
        df
    )

    banner(
        "IDENTIFIER AUDIT"
    )

    for key, value in ids.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    label_df, global_counts = label_distribution(
        df,
        targets
    )

    label_df.to_csv(
        LABEL_SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Missing audit
    # --------------------------------------------------------

    missing_df = missing_audit(
        df,
        targets
    )

    missing_df.to_csv(
        MISSING_SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------

    completeness = completeness_audit(
        df,
        targets
    )

    banner(
        "CASE COMPLETENESS"
    )

    print(
        f"Total studies     : "
        f"{completeness['total_studies']}"
    )

    print(
        f"Complete studies  : "
        f"{completeness['complete_studies']}"
    )

    print(
        f"Incomplete studies: "
        f"{completeness['incomplete_studies']}"
    )

    print(
        f"Complete %        : "
        f"{completeness['complete_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Condition summary
    # --------------------------------------------------------

    condition_df = condition_summary(
        label_df
    )

    # --------------------------------------------------------
    # Study summary
    # --------------------------------------------------------

    if "study_id" in df.columns:

        study_summary = pd.DataFrame(
            {
                "study_id": df["study_id"],
            }
        )

        study_summary[
            "has_complete_labels"
        ] = (
            df[targets]
            .notna()
            .all(axis=1)
            .values
        )

        study_summary.to_csv(
            STUDY_SUMMARY_CSV,
            index=False
        )

    # --------------------------------------------------------
    # Console label distribution
    # --------------------------------------------------------

    banner(
        "LABEL DISTRIBUTION"
    )

    for target in targets:

        print(
            f"\n{target}"
        )

        subset = label_df[
            label_df["target"] == target
        ]

        for _, row in subset.iterrows():

            print(
                f"  "
                f"{row['label']:12s}"
                f" : "
                f"{int(row['count']):5d}"
                f" "
                f"({float(row['percentage']):6.2f}%)"
            )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = create_report(
        df,
        targets,
        missing_targets,
        ids,
        completeness,
        label_df,
        missing_df,
        condition_df
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    summary = {
        "part": 86,
        "title":
            "RSNA Disease Classification Dataset Audit",

        "dataset_root":
            str(RSNA_ROOT),

        "train_csv":
            str(TRAIN_CSV),

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "expected_target_count":
            25,

        "detected_target_count":
            len(targets),

        "detected_targets":
            targets,

        "missing_expected_targets":
            missing_targets,

        "identifier_audit":
            ids,

        "completeness":
            completeness,

        "global_label_counts": {
            str(k): int(v)
            for k, v in global_counts.items()
        },

        "outputs": {
            "label_distribution_csv":
                str(LABEL_SUMMARY_CSV),

            "missing_values_csv":
                str(MISSING_SUMMARY_CSV),

            "study_summary_csv":
                str(STUDY_SUMMARY_CSV),

            "report_txt":
                str(REPORT_TXT),
        },

        "training_performed":
            False,

        "segmentation_modified":
            False,
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    banner(
        "PART 86 COMPLETE"
    )

    print(
        f"Dataset rows      : {len(df)}"
    )

    print(
        f"Detected targets  : "
        f"{len(targets)} / 25"
    )

    print(
        f"Complete studies  : "
        f"{completeness['complete_studies']}"
    )

    print(
        f"Incomplete studies: "
        f"{completeness['incomplete_studies']}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        LABEL_SUMMARY_CSV
    )

    print(
        "\nMissing-value report:"
    )

    print(
        MISSING_SUMMARY_CSV
    )

    print(
        "\nStudy summary:"
    )

    print(
        STUDY_SUMMARY_CSV
    )

    print(
        "\nText report:"
    )

    print(
        REPORT_TXT
    )

    print(
        "\nJSON summary:"
    )

    print(
        SUMMARY_JSON
    )

    print(
        "\nNo model training performed."
    )

    print(
        "Part 84/85 segmentation remains unchanged."
    )

    print(
        "\nSTATUS: DATASET AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()