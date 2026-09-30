from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PART 2.12
# RSNA POINT-SUPERVISION AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part212_point_supervision_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def locate_file(name):
    matches = list(DATASET_ROOT.rglob(name))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {name} inside:\n{DATASET_ROOT}"
        )

    return matches[0]


def main():

    print("=" * 70)
    print("PART 2.12 — RSNA POINT-SUPERVISION AUDIT")
    print("=" * 70)

    train_path = locate_file("train.csv")
    series_path = locate_file("train_series_descriptions.csv")
    coord_path = locate_file("train_label_coordinates.csv")

    print("\nDataset files:")
    print("train.csv:", train_path)
    print("series descriptions:", series_path)
    print("label coordinates:", coord_path)

    train_df = pd.read_csv(train_path)
    series_df = pd.read_csv(series_path)
    coord_df = pd.read_csv(coord_path)

    print("\nShapes")
    print("-" * 70)
    print("train.csv:", train_df.shape)
    print("series descriptions:", series_df.shape)
    print("label coordinates:", coord_df.shape)

    print("\ntrain.csv columns:")
    print(train_df.columns.tolist())

    print("\nseries description columns:")
    print(series_df.columns.tolist())

    print("\ncoordinate columns:")
    print(coord_df.columns.tolist())

    # --------------------------------------------------------
    # Basic identifiers
    # --------------------------------------------------------

    print("\nIdentifier audit")
    print("-" * 70)

    for name, df in [
        ("train", train_df),
        ("series", series_df),
        ("coordinates", coord_df),
    ]:

        print(
            f"{name}: "
            f"study_id={('study_id' in df.columns)}, "
            f"series_id={('series_id' in df.columns)}"
        )

    # --------------------------------------------------------
    # Coordinate audit
    # --------------------------------------------------------

    print("\nCoordinate audit")
    print("-" * 70)

    for column in ["x", "y"]:

        if column in coord_df.columns:

            values = pd.to_numeric(
                coord_df[column],
                errors="coerce"
            )

            print(
                f"{column}: "
                f"valid={values.notna().sum()}, "
                f"missing={values.isna().sum()}, "
                f"min={values.min()}, "
                f"max={values.max()}"
            )

    # --------------------------------------------------------
    # Condition audit
    # --------------------------------------------------------

    condition_columns = [
        c for c in train_df.columns
        if c != "study_id"
    ]

    print("\nTraining-label columns")
    print("-" * 70)

    for c in condition_columns:
        print(c)

    # --------------------------------------------------------
    # Coordinate conditions
    # --------------------------------------------------------

    if "condition" in coord_df.columns:

        print("\nCoordinate conditions")
        print("-" * 70)

        condition_counts = (
            coord_df["condition"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print(condition_counts)

        condition_counts.to_csv(
            OUTPUT_DIR / "coordinate_condition_counts.csv"
        )

    # --------------------------------------------------------
    # Level audit
    # --------------------------------------------------------

    level_columns = [
        c for c in coord_df.columns
        if "level" in c.lower()
    ]

    print("\nPossible level columns:")
    print(level_columns)

    if level_columns:

        for column in level_columns:

            print(f"\n{column} values:")

            print(
                coord_df[column]
                .value_counts(dropna=False)
                .head(30)
            )

    # --------------------------------------------------------
    # Series linkage
    # --------------------------------------------------------

    print("\nSeries linkage audit")
    print("-" * 70)

    if (
        "study_id" in coord_df.columns
        and "series_id" in coord_df.columns
        and "study_id" in series_df.columns
        and "series_id" in series_df.columns
    ):

        coord_keys = (
            coord_df[
                ["study_id", "series_id"]
            ]
            .drop_duplicates()
        )

        series_keys = (
            series_df[
                ["study_id", "series_id"]
            ]
            .drop_duplicates()
        )

        merged = coord_keys.merge(
            series_keys,
            on=["study_id", "series_id"],
            how="left",
            indicator=True
        )

        matched = (
            merged["_merge"] == "both"
        ).sum()

        unmatched = (
            merged["_merge"] != "both"
        ).sum()

        print("Coordinate series:", len(coord_keys))
        print("Matched series:", matched)
        print("Unmatched series:", unmatched)

        merged.to_csv(
            OUTPUT_DIR
            / "coordinate_series_linkage.csv",
            index=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "train_rows": int(len(train_df)),
        "series_rows": int(len(series_df)),
        "coordinate_rows": int(len(coord_df)),
        "coordinate_conditions": (
            int(coord_df["condition"].nunique())
            if "condition" in coord_df.columns
            else None
        ),
        "dataset_root": str(DATASET_ROOT),
        "status": "AUDIT_ONLY",
        "overlay_enabled": False,
    }

    with open(
        OUTPUT_DIR / "part212_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("PART 2.12 COMPLETE")
    print("=" * 70)

    print("\nThis stage ONLY audits the available supervision.")
    print("No fabricated voxel ground truth was created.")
    print("No disease overlay was enabled.")
    print("No checkpoint was modified.")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()