from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# PART 2.13
# RSNA POINT-SUPERVISION MANIFEST
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part213_point_supervision_manifest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Exact model volume established by the project.
MODEL_SHAPE = (64, 96, 96)

# Six-channel model contract.
CLASS_NAME_TO_ID = {
    "Spinal Canal Stenosis": 1,
    "Left Neural Foraminal Narrowing": 2,
    "Right Neural Foraminal Narrowing": 3,
    "Left Subarticular Stenosis": 4,
    "Right Subarticular Stenosis": 5,
}

CLASS_ID_TO_NAME = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


# ============================================================
# FILE DISCOVERY
# ============================================================

def locate_file(name: str) -> Path:

    matches = list(DATASET_ROOT.rglob(name))

    if not matches:
        raise FileNotFoundError(
            f"Could not locate {name} under:\n{DATASET_ROOT}"
        )

    return matches[0]


# ============================================================
# COORDINATE MAPPING
# ============================================================

def resize_coordinate(
    x: float,
    y: float,
    z: float,
    native_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    """
    Map a voxel-center coordinate from native DICOM volume
    into the model volume using the same align_corners=False
    convention used by the established preprocessing.

    native volume:
        (z, y, x)

    model volume:
        (z, y, x)
    """

    nz, nh, nw = [float(v) for v in native_shape]
    tz, th, tw = [float(v) for v in target_shape]

    xo = (float(x) + 0.5) * tw / nw - 0.5
    yo = (float(y) + 0.5) * th / nh - 0.5
    zo = (float(z) + 0.5) * tz / nz - 0.5

    return zo, yo, xo


# ============================================================
# LOAD DICOM SERIES
# ============================================================

def load_part11():

    part11_candidates = [
        ROOT / "src"
        / "segmentation_rsna_part11_controlled_pilot_training_corrected.py",

        ROOT / "src"
        / "segmentation_rsna_part11_controlled_pilot_training.py",
    ]

    part11_path = None

    for path in part11_candidates:

        if path.exists():
            part11_path = path
            break

    if part11_path is None:
        raise FileNotFoundError(
            "Corrected Part 11 loader was not found."
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "part213_part11",
        str(part11_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import {part11_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module



# ============================================================
# MAIN MAPPING
# ============================================================

def build_manifest():

    print("=" * 78)
    print("PART 2.13 — RSNA POINT-SUPERVISION MANIFEST")
    print("=" * 78)

    train_path = locate_file("train.csv")
    series_path = locate_file("train_series_descriptions.csv")
    coord_path = locate_file("train_label_coordinates.csv")

    train_df = pd.read_csv(train_path)
    series_df = pd.read_csv(series_path)
    coord_df = pd.read_csv(coord_path)

    print("\nDataset:")
    print("  train.csv                 :", train_df.shape)
    print("  series descriptions      :", series_df.shape)
    print("  coordinate annotations   :", coord_df.shape)

    # --------------------------------------------------------
    # Validate coordinate schema
    # --------------------------------------------------------

    required = {
        "study_id",
        "series_id",
        "instance_number",
        "condition",
        "level",
        "x",
        "y",
    }

    missing = required - set(coord_df.columns)

    if missing:
        raise RuntimeError(
            f"Missing coordinate columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Normalize types
    # --------------------------------------------------------

    coord_df["study_id"] = coord_df["study_id"].astype(str)
    coord_df["series_id"] = coord_df["series_id"].astype(str)
    coord_df["condition"] = coord_df["condition"].astype(str).str.strip()
    coord_df["level"] = coord_df["level"].astype(str).str.strip()

    coord_df["instance_number"] = pd.to_numeric(
        coord_df["instance_number"],
        errors="coerce",
    )

    coord_df["x"] = pd.to_numeric(
        coord_df["x"],
        errors="coerce",
    )

    coord_df["y"] = pd.to_numeric(
        coord_df["y"],
        errors="coerce",
    )

    coord_df = coord_df.dropna(
        subset=[
            "study_id",
            "series_id",
            "instance_number",
            "x",
            "y",
        ]
    ).copy()

    # --------------------------------------------------------
    # Keep only documented five foreground conditions
    # --------------------------------------------------------

    before = len(coord_df)

    coord_df = coord_df[
        coord_df["condition"].isin(
            CLASS_NAME_TO_ID.keys()
        )
    ].copy()

    after = len(coord_df)

    print("\nCondition filtering:")
    print("  Original annotations :", before)
    print("  Accepted annotations :", after)
    print("  Removed              :", before - after)

    # --------------------------------------------------------
    # Load established loaders
    # --------------------------------------------------------

    print("\nLoading established Part 11 DICOM utilities...")

    p11 = load_part11()

    print("  Part 11: PASS")
    print("  Pseudo-mask loading: NOT USED")

    # --------------------------------------------------------
    # Use a deterministic subset first.
    #
    # This is intentionally NOT training.
    # It verifies mapping before generating a full manifest.
    # --------------------------------------------------------

    series_keys = (
        coord_df[
            ["study_id", "series_id"]
        ]
        .drop_duplicates()
        .sort_values(
            ["study_id", "series_id"]
        )
    )

    print(
        "\nUnique annotated series:",
        len(series_keys),
    )

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    mapped_rows: List[Dict[str, Any]] = []

    failed_series = []
    total_annotations = 0
    mapped_annotations = 0

    for series_no, (_, key) in enumerate(
        series_keys.iterrows(),
        start=1,
    ):

        study_id = str(key["study_id"])
        series_id = str(key["series_id"])

        # Find the corresponding series row.
        candidates = series_df[
            (series_df["study_id"].astype(str) == study_id)
            & (series_df["series_id"].astype(str) == series_id)
        ]

        if candidates.empty:

            failed_series.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "reason": "series_description_not_found",
                }
            )

            continue

        # The same series may have one description row.
        row = candidates.iloc[0]

        # ----------------------------------------------------
        # Get annotations for this series.
        # ----------------------------------------------------

        ann = coord_df[
            (coord_df["study_id"] == study_id)
            & (coord_df["series_id"] == series_id)
        ].copy()

        total_annotations += len(ann)

        # ----------------------------------------------------
        # Create a minimal Part11-compatible row.
        # ----------------------------------------------------

        series_row = pd.Series(
            {
                "study_id": study_id,
                "series_id": series_id,
            }
        )

                # ----------------------------------------------------
        # DICOM-ONLY LOAD
        #
        # IMPORTANT:
        # This step intentionally does NOT call
        # load_tensor_case().
        #
        # We are auditing RSNA point coordinates, so a
        # pseudo-mask is neither required nor used.
        # ----------------------------------------------------

        try:

            series_dir = p11.resolve_series_dir(
                series_row
            )

            image_native, dicom_records, dicom_info = (
                p11.read_dicom_series_robust(
                    series_dir
                )
            )

            image_native = np.asarray(
                image_native,
                dtype=np.float32,
            )

            native_shape = tuple(
                int(v)
                for v in image_native.shape
            )

        except Exception as exc:

            failed_series.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "reason": (
                        "dicom_mapping_error: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # InstanceNumber -> native z
        # ----------------------------------------------------

        instance_to_z = {}

        for z, ds in enumerate(dicom_records):

            instance = ds.get(
                "InstanceNumber",
                None,
            )

            if instance is None:
                continue

            try:
                instance = int(instance)
            except Exception:
                continue

            instance_to_z[instance] = int(z)

        # ----------------------------------------------------
        # Map every RSNA point.
        # ----------------------------------------------------

        for _, a in ann.iterrows():

            try:

                instance = int(
                    float(a["instance_number"])
                )

                x = float(a["x"])
                y = float(a["y"])

            except Exception:

                continue

            condition = str(
                a["condition"]
            ).strip()

            class_id = CLASS_NAME_TO_ID.get(
                condition
            )

            if class_id is None:
                continue

            if instance not in instance_to_z:
                continue

            native_z = instance_to_z[instance]

            # Native image is represented as:
            # (z, y, x)
            if not (
                0 <= x < native_shape[2]
                and
                0 <= y < native_shape[1]
                and
                0 <= native_z < native_shape[0]
            ):
                continue

            model_z, model_y, model_x = (
                resize_coordinate(
                    x=x,
                    y=y,
                    z=native_z,
                    native_shape=native_shape,
                    target_shape=MODEL_SHAPE,
                )
            )

            zi = int(round(model_z))
            yi = int(round(model_y))
            xi = int(round(model_x))

            if not (
                0 <= zi < MODEL_SHAPE[0]
                and
                0 <= yi < MODEL_SHAPE[1]
                and
                0 <= xi < MODEL_SHAPE[2]
            ):
                continue

            mapped_annotations += 1

            mapped_rows.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "series_description": str(
                        row.get(
                            "series_description",
                            ""
                        )
                    ),
                    "condition": condition,
                    "class_id": int(class_id),
                    "class_name": CLASS_ID_TO_NAME[
                        class_id
                    ],
                    "level": str(
                        a["level"]
                    ),
                    "instance_number": instance,
                    "native_z": int(native_z),
                    "native_y": float(y),
                    "native_x": float(x),
                    "model_z_float": float(model_z),
                    "model_y_float": float(model_y),
                    "model_x_float": float(model_x),
                    "model_z": int(zi),
                    "model_y": int(yi),
                    "model_x": int(xi),
                    "native_shape": str(
                        native_shape
                    ),
                    "model_shape": str(
                        MODEL_SHAPE
                    ),
                }
            )

        if (
            series_no == 1
            or series_no % 250 == 0
            or series_no == len(series_keys)
        ):

            print(
                f"  Processed "
                f"{series_no}/{len(series_keys)} "
                f"series | "
                f"mapped points={mapped_annotations}"
            )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest_df = pd.DataFrame(
        mapped_rows
    )

    failed_df = pd.DataFrame(
        failed_series
    )

    manifest_path = (
        OUTPUT_DIR
        / "part213_point_manifest.csv"
    )

    failed_path = (
        OUTPUT_DIR
        / "part213_failed_series.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    failed_df.to_csv(
        failed_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("PART 2.13 MAPPING SUMMARY")
    print("=" * 78)

    print(
        "Annotated series        :",
        len(series_keys),
    )

    print(
        "Input annotation points :",
        total_annotations,
    )

    print(
        "Mapped annotation points:",
        mapped_annotations,
    )

    print(
        "Mapping success rate    :",
        f"{100.0 * mapped_annotations / max(total_annotations, 1):.3f}%",
    )

    print(
        "Failed series           :",
        len(failed_series),
    )

    if not manifest_df.empty:

        print("\nMapped points by class:")

        class_counts = (
            manifest_df[
                ["class_id", "class_name"]
            ]
            .drop_duplicates()
        )

        counts = (
            manifest_df
            .groupby(
                ["class_id", "class_name"]
            )
            .size()
            .reset_index(
                name="mapped_points"
            )
            .sort_values("class_id")
        )

        print(
            counts.to_string(
                index=False
            )
        )

        counts.to_csv(
            OUTPUT_DIR
            / "part213_class_counts.csv",
            index=False,
        )

        print("\nMapped points by level:")

        level_counts = (
            manifest_df["level"]
            .value_counts()
            .sort_index()
        )

        print(level_counts)

        level_counts.to_csv(
            OUTPUT_DIR
            / "part213_level_counts.csv"
        )

        print("\nMapped model-grid bounds:")

        for axis in [
            "model_z",
            "model_y",
            "model_x",
        ]:

            print(
                f"  {axis}: "
                f"{manifest_df[axis].min()} -> "
                f"{manifest_df[axis].max()}"
            )

    # --------------------------------------------------------
    # Scientific status
    # --------------------------------------------------------

    summary = {
        "part": "2.13",
        "status": "COMPLETE",
        "model_shape": list(MODEL_SHAPE),
        "annotated_series": int(
            len(series_keys)
        ),
        "input_annotation_points": int(
            total_annotations
        ),
        "mapped_annotation_points": int(
            mapped_annotations
        ),
        "mapping_success_rate_percent": float(
            100.0
            * mapped_annotations
            / max(total_annotations, 1)
        ),
        "failed_series": int(
            len(failed_series)
        ),
        "class_contract": CLASS_ID_TO_NAME,
        "target_type": (
            "point_supervision_coordinates"
        ),
        "manual_voxel_segmentation_ground_truth": False,
        "training_performed": False,
        "checkpoint_modified": False,
        "disease_overlay_enabled": False,
        "scientific_limitation": (
            "RSNA coordinates are point/localization "
            "annotations, not manual voxel-wise "
            "segmentation masks."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "part213_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nOutputs:")
    print("  ", manifest_path)
    print("  ", failed_path)
    print(
        "  ",
        OUTPUT_DIR
        / "part213_class_counts.csv"
    )
    print(
        "  ",
        OUTPUT_DIR
        / "part213_level_counts.csv"
    )
    print("  ", summary_path)

    print("\n" + "=" * 78)
    print("PART 2.13 COMPLETE")
    print("=" * 78)

    print(
        "\nNo model training was performed."
    )
    print(
        "No checkpoint was modified."
    )
    print(
        "No disease overlay was enabled."
    )
    print(
        "The manifest contains point supervision only."
    )


if __name__ == "__main__":
    build_manifest()