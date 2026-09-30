from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd


# ============================================================
# PART 87
# LEAKAGE-FREE RSNA CLASSIFICATION SPLIT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_CSV = RSNA_ROOT / "train.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"
TRAIN_IMAGES = RSNA_ROOT / "train_images"


OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
)

REPORT_DIR = ROOT / "reports"

TRAIN_MANIFEST = (
    OUTPUT_DIR
    / "part87_train_manifest.csv"
)

VAL_MANIFEST = (
    OUTPUT_DIR
    / "part87_validation_manifest.csv"
)

TRAIN_STUDIES = (
    OUTPUT_DIR
    / "part87_train_study_ids.csv"
)

VAL_STUDIES = (
    OUTPUT_DIR
    / "part87_validation_study_ids.csv"
)

LEAKAGE_CSV = (
    OUTPUT_DIR
    / "part87_study_overlap_audit.csv"
)

SERIES_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part87_series_selection_summary.csv"
)

TARGET_SUMMARY_CSV = (
    OUTPUT_DIR
    / "part87_split_target_summary.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part87_classification_split_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part87_classification_split_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_FRACTION = 0.80

PRIMARY_SERIES = "Sagittal T2/STIR"

FALLBACK_SERIES = [
    "Sagittal T1",
    "Axial T2",
]

EXPECTED_SERIES = [
    "Axial T2",
    "Sagittal T1",
    "Sagittal T2/STIR",
]


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


TARGET_COLUMNS = [
    f"{condition}_{level}"
    for condition in CONDITIONS
    for level in LEVELS
]


# ============================================================
# UTILITIES
# ============================================================

def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)


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


def normalize_series_name(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_percentage(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        float(numerator)
        / float(denominator)
        * 100.0
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    banner("PART 87 PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "train.csv": TRAIN_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
        "train_images": TRAIN_IMAGES,
    }

    missing = []

    for name, path in paths.items():

        exists = path.exists()

        print(
            f"{name:<38}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            missing.append(str(path))

    if missing:

        raise FileNotFoundError(
            "\nMissing required input(s):\n"
            + "\n".join(missing)
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_metadata():

    banner("LOADING RSNA METADATA")

    train = pd.read_csv(
        TRAIN_CSV
    )

    series = pd.read_csv(
        SERIES_CSV
    )

    train.columns = [
        normalize_column_name(c)
        for c in train.columns
    ]

    series.columns = [
        normalize_column_name(c)
        for c in series.columns
    ]

    print(
        f"train.csv rows                  : {len(train)}"
    )

    print(
        f"train.csv columns               : {len(train.columns)}"
    )

    print(
        f"train_series_descriptions rows  : {len(series)}"
    )

    required_train = {
        "study_id"
    }.union(
        TARGET_COLUMNS
    )

    required_series = {
        "study_id",
        "series_id",
        "series_description",
    }

    missing_train = (
        required_train
        - set(train.columns)
    )

    missing_series = (
        required_series
        - set(series.columns)
    )

    if missing_train:

        raise RuntimeError(
            "Missing train.csv columns:\n"
            + "\n".join(
                sorted(missing_train)
            )
        )

    if missing_series:

        raise RuntimeError(
            "Missing series metadata columns:\n"
            + "\n".join(
                sorted(missing_series)
            )
        )

    return train, series


# ============================================================
# STUDY ID NORMALIZATION
# ============================================================

def normalize_study_ids(
    train,
    series
):

    train = train.copy()
    series = series.copy()

    train["study_id"] = (
        train["study_id"]
        .astype(str)
        .str.strip()
    )

    series["study_id"] = (
        series["study_id"]
        .astype(str)
        .str.strip()
    )

    series["series_id"] = (
        series["series_id"]
        .astype(str)
        .str.strip()
    )

    return train, series


# ============================================================
# SERIES SELECTION
# ============================================================

def series_priority(series_description):

    name = normalize_series_name(
        series_description
    )

    if name == PRIMARY_SERIES:
        return 0

    if name == FALLBACK_SERIES[0]:
        return 1

    if name == FALLBACK_SERIES[1]:
        return 2

    return 99


def series_directory_exists(
    study_id,
    series_id
):

    path = (
        TRAIN_IMAGES
        / str(study_id)
        / str(series_id)
    )

    return path.exists() and path.is_dir()


def count_series_files(
    study_id,
    series_id
):

    path = (
        TRAIN_IMAGES
        / str(study_id)
        / str(series_id)
    )

    if not path.exists():

        return 0

    try:

        return sum(
            1
            for p in path.iterdir()
            if p.is_file()
        )

    except Exception:

        return 0


def select_best_series(
    study_id,
    study_series
):

    if study_series.empty:

        return {
            "selected": False,
            "series_id": "",
            "series_description": "",
            "selection_rank": None,
            "series_path": "",
            "series_exists": False,
            "dicom_file_count": 0,
            "selection_reason": "NO_SERIES_METADATA",
        }

    candidates = []

    for _, row in study_series.iterrows():

        description = normalize_series_name(
            row["series_description"]
        )

        rank = series_priority(
            description
        )

        if rank >= 99:
            continue

        series_id = str(
            row["series_id"]
        ).strip()

        exists = series_directory_exists(
            study_id,
            series_id
        )

        file_count = count_series_files(
            study_id,
            series_id
        )

        candidates.append(
            {
                "series_id": series_id,
                "series_description": description,
                "selection_rank": rank,
                "series_exists": exists,
                "dicom_file_count": file_count,
            }
        )

    if not candidates:

        return {
            "selected": False,
            "series_id": "",
            "series_description": "",
            "selection_rank": None,
            "series_path": "",
            "series_exists": False,
            "dicom_file_count": 0,
            "selection_reason": "NO_SUPPORTED_SERIES",
        }

    # Prefer:
    # 1. Series type priority
    # 2. Existing directory
    # 3. Larger DICOM file count
    candidates.sort(
        key=lambda x: (
            x["selection_rank"],
            not x["series_exists"],
            -x["dicom_file_count"],
        )
    )

    best = candidates[0]

    if not best["series_exists"]:

        return {
            "selected": False,
            "series_id": best["series_id"],
            "series_description":
                best["series_description"],
            "selection_rank":
                best["selection_rank"],
            "series_path": str(
                TRAIN_IMAGES
                / str(study_id)
                / best["series_id"]
            ),
            "series_exists": False,
            "dicom_file_count":
                best["dicom_file_count"],
            "selection_reason":
                "SUPPORTED_SERIES_FOUND_BUT_DIRECTORY_MISSING",
        }

    if (
        best["series_description"]
        == PRIMARY_SERIES
    ):

        reason = "PRIMARY_SAGITTAL_T2_STIR"

    elif (
        best["series_description"]
        == FALLBACK_SERIES[0]
    ):

        reason = "FALLBACK_SAGITTAL_T1"

    elif (
        best["series_description"]
        == FALLBACK_SERIES[1]
    ):

        reason = "FALLBACK_AXIAL_T2"

    else:

        reason = "OTHER"

    return {
        "selected": True,
        "series_id": best["series_id"],
        "series_description":
            best["series_description"],
        "selection_rank":
            best["selection_rank"],
        "series_path": str(
            TRAIN_IMAGES
            / str(study_id)
            / best["series_id"]
        ),
        "series_exists": True,
        "dicom_file_count":
            best["dicom_file_count"],
        "selection_reason": reason,
    }


# ============================================================
# BUILD SERIES MANIFEST
# ============================================================

def build_series_manifest(
    train,
    series
):

    banner(
        "BUILDING STUDY-LEVEL MRI SERIES MANIFEST"
    )

    records = []

    grouped = series.groupby(
        "study_id"
    )

    for i, study_id in enumerate(
        train["study_id"].unique(),
        start=1
    ):

        if i % 250 == 0 or i == 1:

            print(
                f"Processing study "
                f"{i}/{train['study_id'].nunique()}..."
            )

        if study_id in grouped.groups:

            study_series = grouped.get_group(
                study_id
            )

        else:

            study_series = pd.DataFrame(
                columns=series.columns
            )

        selected = select_best_series(
            study_id,
            study_series
        )

        records.append(
            {
                "study_id": study_id,
                **selected,
            }
        )

    manifest = pd.DataFrame(
        records
    )

    return manifest


# ============================================================
# STUDY-LEVEL SPLIT
# ============================================================

def create_study_split(
    study_ids,
    seed=42,
    train_fraction=0.80
):

    study_ids = list(
        map(str, study_ids)
    )

    rng = np.random.default_rng(
        seed
    )

    shuffled = np.array(
        study_ids,
        dtype=object
    )

    rng.shuffle(
        shuffled
    )

    train_count = int(
        round(
            len(shuffled)
            * train_fraction
        )
    )

    train_ids = shuffled[
        :train_count
    ].tolist()

    val_ids = shuffled[
        train_count:
    ].tolist()

    return (
        train_ids,
        val_ids
    )


# ============================================================
# SPLIT MANIFEST
# ============================================================

def create_split_manifests(
    train,
    series_manifest,
    train_ids,
    val_ids
):

    labels = train.copy()

    labels["split"] = ""

    train_set = set(
        train_ids
    )

    val_set = set(
        val_ids
    )

    labels.loc[
        labels["study_id"].isin(
            train_set
        ),
        "split"
    ] = "train"

    labels.loc[
        labels["study_id"].isin(
            val_set
        ),
        "split"
    ] = "validation"

    merged = labels.merge(
        series_manifest,
        on="study_id",
        how="left",
        validate="one_to_one"
    )

    train_manifest = merged[
        merged["split"] == "train"
    ].copy()

    val_manifest = merged[
        merged["split"] == "validation"
    ].copy()

    return (
        train_manifest,
        val_manifest,
        merged
    )


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def leakage_audit(
    train_ids,
    val_ids
):

    train_set = set(
        map(str, train_ids)
    )

    val_set = set(
        map(str, val_ids)
    )

    overlap = sorted(
        train_set.intersection(
            val_set
        )
    )

    return overlap


# ============================================================
# TARGET SUMMARY
# ============================================================

def target_split_summary(
    train_manifest,
    val_manifest
):

    rows = []

    for target in TARGET_COLUMNS:

        train_counts = (
            train_manifest[target]
            .value_counts(
                dropna=False
            )
        )

        val_counts = (
            val_manifest[target]
            .value_counts(
                dropna=False
            )
        )

        labels = sorted(
            set(
                train_counts.index.tolist()
                + val_counts.index.tolist()
            ),
            key=lambda x: str(x)
        )

        for label in labels:

            train_count = int(
                train_counts.get(
                    label,
                    0
                )
            )

            val_count = int(
                val_counts.get(
                    label,
                    0
                )
            )

            rows.append(
                {
                    "target": target,
                    "label": str(label),
                    "train_count":
                        train_count,
                    "validation_count":
                        val_count,
                    "train_percentage":
                        safe_percentage(
                            train_count,
                            len(train_manifest)
                        ),
                    "validation_percentage":
                        safe_percentage(
                            val_count,
                            len(val_manifest)
                        ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SERIES SUMMARY
# ============================================================

def series_summary(
    manifest
):

    selected = manifest[
        manifest["selected"] == True
    ].copy()

    counts = (
        selected[
            "series_description"
        ]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "series_description",
        "study_count"
    ]

    counts["percentage"] = (
        counts["study_count"]
        / len(manifest)
        * 100.0
    )

    return counts


# ============================================================
# WRITE REPORT
# ============================================================

def write_report(
    train,
    series,
    series_manifest,
    train_manifest,
    val_manifest,
    overlap,
    target_summary,
    train_ids,
    val_ids
):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "PART 87 — LEAKAGE-FREE RSNA CLASSIFICATION SPLIT"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        f"Project root: {ROOT}"
    )

    lines.append(
        f"RSNA root: {RSNA_ROOT}"
    )

    lines.append("")

    lines.append(
        "DATASET"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Total studies: {len(train)}"
    )

    lines.append(
        f"Unique studies: "
        f"{train['study_id'].nunique()}"
    )

    lines.append(
        f"Classification targets: "
        f"{len(TARGET_COLUMNS)}"
    )

    lines.append("")

    lines.append(
        "STUDY-LEVEL SPLIT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Random seed: {SEED}"
    )

    lines.append(
        f"Training fraction: "
        f"{TRAIN_FRACTION:.2f}"
    )

    lines.append(
        f"Training studies: {len(train_ids)}"
    )

    lines.append(
        f"Validation studies: {len(val_ids)}"
    )

    lines.append(
        f"Training percentage: "
        f"{safe_percentage(len(train_ids), len(train)):.2f}%"
    )

    lines.append(
        f"Validation percentage: "
        f"{safe_percentage(len(val_ids), len(train)):.2f}%"
    )

    lines.append("")

    lines.append(
        "LEAKAGE AUDIT"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Overlapping studies: {len(overlap)}"
    )

    if overlap:

        lines.append(
            "FAIL — Study-level leakage detected."
        )

        for study_id in overlap[:20]:

            lines.append(
                f"  {study_id}"
            )

    else:

        lines.append(
            "PASS — No study appears in both "
            "training and validation."
        )

    lines.append("")

    lines.append(
        "MRI SERIES SELECTION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Primary series: {PRIMARY_SERIES}"
    )

    lines.append(
        f"Fallback 1: {FALLBACK_SERIES[0]}"
    )

    lines.append(
        f"Fallback 2: {FALLBACK_SERIES[1]}"
    )

    total = len(series_manifest)

    selected = int(
        series_manifest["selected"].sum()
    )

    missing = total - selected

    lines.append(
        f"Studies with selected MRI series: "
        f"{selected}"
    )

    lines.append(
        f"Studies without usable selected series: "
        f"{missing}"
    )

    lines.append(
        f"Series coverage: "
        f"{safe_percentage(selected, total):.2f}%"
    )

    lines.append("")

    lines.append(
        "SELECTED SERIES DISTRIBUTION"
    )

    lines.append(
        "-" * 78
    )

    distribution = series_summary(
        series_manifest
    )

    for _, row in distribution.iterrows():

        lines.append(
            f"{row['series_description']:25s}"
            f" : "
            f"{int(row['study_count']):5d}"
            f" "
            f"({float(row['percentage']):6.2f}%)"
        )

    lines.append("")

    lines.append(
        "CLASSIFICATION SPLIT SUMMARY"
    )

    lines.append(
        "-" * 78
    )

    # Summarize only the three expected severity
    # labels and missing values.
    for target in TARGET_COLUMNS:

        subset = target_summary[
            target_summary["target"]
            == target
        ]

        lines.append(
            f"\n{target}"
        )

        for _, row in subset.iterrows():

            lines.append(
                f"  "
                f"{row['label']:12s}"
                f" | train="
                f"{int(row['train_count']):5d}"
                f" | val="
                f"{int(row['validation_count']):5d}"
            )

    lines.append("")

    lines.append(
        "FINAL STATUS"
    )

    lines.append(
        "-" * 78
    )

    if len(overlap) == 0:

        lines.append(
            "PASS — Study-level leakage-free split."
        )

    else:

        lines.append(
            "FAIL — Study overlap must be resolved."
        )

    if selected == total:

        lines.append(
            "PASS — Every study has a usable "
            "selected MRI series."
        )

    else:

        lines.append(
            "REVIEW — Some studies have no usable "
            "selected MRI series."
        )

    lines.append("")

    lines.append(
        "Part 87 does not train a neural network."
    )

    lines.append(
        "Part 84/85 segmentation files are unchanged."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 87 — LEAKAGE-FREE RSNA CLASSIFICATION SPLIT"
    )

    print(
        f"Project root:\n{ROOT}"
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    train, series = load_metadata()

    train, series = normalize_study_ids(
        train,
        series
    )

    # --------------------------------------------------------
    # Verify unique studies
    # --------------------------------------------------------

    banner(
        "STUDY ID VALIDATION"
    )

    total_rows = len(train)

    unique_studies = (
        train["study_id"]
        .nunique()
    )

    duplicate_rows = int(
        train["study_id"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    print(
        f"Rows            : {total_rows}"
    )

    print(
        f"Unique studies  : {unique_studies}"
    )

    print(
        f"Duplicate rows  : {duplicate_rows}"
    )

    if total_rows != unique_studies:

        raise RuntimeError(
            "train.csv does not contain one row per study."
        )

    # --------------------------------------------------------
    # Build MRI series manifest
    # --------------------------------------------------------

    series_manifest = build_series_manifest(
        train,
        series
    )

    # --------------------------------------------------------
    # Create study-level split
    # --------------------------------------------------------

    banner(
        "CREATING STUDY-LEVEL 80/20 SPLIT"
    )

    train_ids, val_ids = create_study_split(
        train["study_id"].tolist(),
        seed=SEED,
        train_fraction=TRAIN_FRACTION
    )

    print(
        f"Training studies   : {len(train_ids)}"
    )

    print(
        f"Validation studies : {len(val_ids)}"
    )

    # --------------------------------------------------------
    # Leakage audit
    # --------------------------------------------------------

    overlap = leakage_audit(
        train_ids,
        val_ids
    )

    print(
        f"Study overlap      : {len(overlap)}"
    )

    if overlap:

        print(
            "\nOVERLAPPING STUDIES:"
        )

        for study_id in overlap[:20]:

            print(
                f"  {study_id}"
            )

        raise RuntimeError(
            "Study-level leakage detected."
        )

    # --------------------------------------------------------
    # Create manifests
    # --------------------------------------------------------

    (
        train_manifest,
        val_manifest,
        complete_manifest
    ) = create_split_manifests(
        train,
        series_manifest,
        train_ids,
        val_ids
    )

    # --------------------------------------------------------
    # Verify manifest counts
    # --------------------------------------------------------

    if len(train_manifest) != len(train_ids):

        raise RuntimeError(
            "Training manifest count mismatch."
        )

    if len(val_manifest) != len(val_ids):

        raise RuntimeError(
            "Validation manifest count mismatch."
        )

    # --------------------------------------------------------
    # Verify no overlap after merge
    # --------------------------------------------------------

    post_train = set(
        train_manifest["study_id"]
    )

    post_val = set(
        val_manifest["study_id"]
    )

    post_overlap = post_train.intersection(
        post_val
    )

    if post_overlap:

        raise RuntimeError(
            "Post-merge study leakage detected."
        )

    # --------------------------------------------------------
    # Series coverage
    # --------------------------------------------------------

    banner(
        "MRI SERIES COVERAGE"
    )

    selected_total = int(
        complete_manifest["selected"].sum()
    )

    missing_total = (
        len(complete_manifest)
        - selected_total
    )

    print(
        f"Total studies with metadata : "
        f"{len(complete_manifest)}"
    )

    print(
        f"Usable selected series      : "
        f"{selected_total}"
    )

    print(
        f"No usable selected series   : "
        f"{missing_total}"
    )

    print(
        f"Coverage                     : "
        f"{safe_percentage(selected_total, len(complete_manifest)):.2f}%"
    )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    train_manifest.to_csv(
        TRAIN_MANIFEST,
        index=False
    )

    val_manifest.to_csv(
        VAL_MANIFEST,
        index=False
    )

    pd.DataFrame(
        {
            "study_id": train_ids
        }
    ).to_csv(
        TRAIN_STUDIES,
        index=False
    )

    pd.DataFrame(
        {
            "study_id": val_ids
        }
    ).to_csv(
        VAL_STUDIES,
        index=False
    )

    pd.DataFrame(
        {
            "study_id": overlap
        }
    ).to_csv(
        LEAKAGE_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Series summary
    # --------------------------------------------------------

    series_summary_df = series_summary(
        complete_manifest
    )

    series_summary_df.to_csv(
        SERIES_SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_summary = target_split_summary(
        train_manifest,
        val_manifest
    )

    target_summary.to_csv(
        TARGET_SUMMARY_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = write_report(
        train,
        series,
        complete_manifest,
        train_manifest,
        val_manifest,
        overlap,
        target_summary,
        train_ids,
        val_ids
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

    train_selected = int(
        train_manifest["selected"].sum()
    )

    val_selected = int(
        val_manifest["selected"].sum()
    )

    summary = {
        "part": 87,
        "title":
            "Leakage-Free RSNA Classification Split",

        "seed": SEED,

        "train_fraction":
            TRAIN_FRACTION,

        "dataset": {
            "rows":
                int(len(train)),

            "unique_studies":
                int(unique_studies),

            "classification_targets":
                len(TARGET_COLUMNS),
        },

        "split": {
            "train_studies":
                len(train_ids),

            "validation_studies":
                len(val_ids),

            "train_percentage":
                safe_percentage(
                    len(train_ids),
                    len(train)
                ),

            "validation_percentage":
                safe_percentage(
                    len(val_ids),
                    len(train)
                ),

            "study_overlap":
                len(overlap),
        },

        "series_selection": {
            "primary":
                PRIMARY_SERIES,

            "fallbacks":
                FALLBACK_SERIES,

            "total_studies":
                len(complete_manifest),

            "selected_series":
                selected_total,

            "missing_series":
                missing_total,

            "coverage_percentage":
                safe_percentage(
                    selected_total,
                    len(complete_manifest)
                ),

            "training_selected":
                train_selected,

            "validation_selected":
                val_selected,
        },

        "target_columns":
            TARGET_COLUMNS,

        "outputs": {
            "train_manifest":
                str(TRAIN_MANIFEST),

            "validation_manifest":
                str(VAL_MANIFEST),

            "train_study_ids":
                str(TRAIN_STUDIES),

            "validation_study_ids":
                str(VAL_STUDIES),

            "leakage_audit":
                str(LEAKAGE_CSV),

            "series_summary":
                str(SERIES_SUMMARY_CSV),

            "target_summary":
                str(TARGET_SUMMARY_CSV),

            "report":
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
    # Final console
    # --------------------------------------------------------

    banner(
        "PART 87 COMPLETE"
    )

    print(
        f"Total studies       : {len(train)}"
    )

    print(
        f"Training studies    : {len(train_ids)}"
    )

    print(
        f"Validation studies  : {len(val_ids)}"
    )

    print(
        f"Study overlap       : {len(overlap)}"
    )

    print(
        f"Selected MRI series : {selected_total}"
    )

    print(
        f"Missing MRI series  : {missing_total}"
    )

    print(
        f"Series coverage     : "
        f"{safe_percentage(selected_total, len(complete_manifest)):.2f}%"
    )

    print(
        "\nTraining manifest:"
    )

    print(
        TRAIN_MANIFEST
    )

    print(
        "\nValidation manifest:"
    )

    print(
        VAL_MANIFEST
    )

    print(
        "\nStudy leakage audit:"
    )

    print(
        LEAKAGE_CSV
    )

    print(
        "\nSeries summary:"
    )

    print(
        SERIES_SUMMARY_CSV
    )

    print(
        "\nTarget summary:"
    )

    print(
        TARGET_SUMMARY_CSV
    )

    print(
        "\nReport:"
    )

    print(
        REPORT_TXT
    )

    print(
        "\nJSON:"
    )

    print(
        SUMMARY_JSON
    )

    print(
        "\nSTATUS: LEAKAGE-FREE STUDY-LEVEL SPLIT CREATED"
    )


if __name__ == "__main__":
    main()