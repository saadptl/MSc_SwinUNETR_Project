"""
PART 4.6B
Multi-Point / Multi-Level Point-Targeted 3D XAI Validation

Purpose
-------
Validate point-targeted 3D XAI against ALL available RSNA annotation
points for the selected study/series.

This experiment:
    - does NOT retrain the model
    - does NOT modify the checkpoint
    - does NOT modify Part 2.20B geometry
    - does NOT modify the dashboard
    - does NOT fabricate voxel-wise ground truth
    - reuses the already validated Part 4.6 implementation

Target:
    decoder1.conv_block.norm2

Study:
    7143189

Series:
    3219733239

Diseases:
    LFNN - class 2
    RFNN - class 3
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part46b_multi_point_3d_xai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

STUDY_ID = "7143189"
SERIES_ID = "3219733239"

TARGET_LAYER_NAME = (
    "decoder1.conv_block.norm2"
)

TARGET_DISEASES = {
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. IMPORT THE WORKING PART 4.6 IMPLEMENTATION
# ============================================================

import segmentation_part46_point_targeted_3d_xai as part46


# ============================================================
# 4. UTILITY
# ============================================================

def safe_float(value):
    """Convert a value to Python float safely."""
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value):
    """Convert a value to Python int safely."""
    try:
        return int(value)
    except Exception:
        return None


def normalize_level(level):
    """
    Convert a level name into a filesystem-safe name.
    """
    return (
        str(level)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


# ============================================================
# 5. LOAD CASE USING THE EXACT PART 2.20B PIPELINE
# ============================================================

def load_case_from_part220b():
    """
    Load the canonical image and all RSNA points using the
    exact Part 2.20B physical-space implementation.
    """

    import segmentation_rsna_part220b_geometry_corrected_training as part220b

    print("\nLoading Part 2.20B module...")

    manifest = part220b.load_manifest()

    print(
        f"Manifest rows: {len(manifest)}"
    )

    case_manifest = manifest[
        (
            manifest["study_id"].astype(str)
            == str(STUDY_ID)
        )
        &
        (
            manifest["series_id"].astype(str)
            == str(SERIES_ID)
        )
    ].copy()

    if case_manifest.empty:
        raise RuntimeError(
            "No manifest rows found for "
            f"{STUDY_ID}/{SERIES_ID}"
        )

    print(
        f"Case manifest rows: "
        f"{len(case_manifest)}"
    )

    result = part220b.load_case(
        STUDY_ID,
        SERIES_ID,
        case_manifest,
    )

    if not isinstance(result, tuple):
        raise RuntimeError(
            "Unexpected load_case() result."
        )

    if len(result) != 3:
        raise RuntimeError(
            "Expected load_case() to return "
            "(canonical, points, geometry)."
        )

    canonical, points, geometry = result

    canonical = np.asarray(
        canonical,
        dtype=np.float32,
    )

    print(
        f"Canonical image shape: "
        f"{canonical.shape}"
    )

    print(
        f"Number of points: "
        f"{len(points)}"
    )

    return (
        canonical,
        points,
        geometry,
    )


# ============================================================
# 6. PRINT ALL ANNOTATION POINTS
# ============================================================

def print_points(points):

    print("\n" + "=" * 78)
    print("ALL CANONICAL RSNA ANNOTATION POINTS")
    print("=" * 78)

    rows = []

    for point in points:

        class_id = safe_int(
            point.get("class_id")
        )

        if class_id not in TARGET_DISEASES:
            continue

        rows.append(
            {
                "class_id": class_id,
                "disease": TARGET_DISEASES[
                    class_id
                ],
                "level": point.get(
                    "level"
                ),
                "z": safe_float(
                    point.get("z")
                ),
                "y": safe_float(
                    point.get("y")
                ),
                "x": safe_float(
                    point.get("x")
                ),
            }
        )

    table = pd.DataFrame(rows)

    if not table.empty:
        print(
            table.to_string(
                index=False
            )
        )

    print("=" * 78)

    return table


# ============================================================
# 7. FIND THE XAI FUNCTION
# ============================================================

def get_xai_function():

    candidates = [
        "compute_point_targeted_xai",
    ]

    for name in candidates:

        function = getattr(
            part46,
            name,
            None,
        )

        if callable(function):
            print(
                f"\nUsing Part 4.6 XAI function: "
                f"{name}"
            )
            return function

    raise AttributeError(
        "Could not find "
        "compute_point_targeted_xai() "
        "inside Part 4.6."
    )


# ============================================================
# 8. CALL EXISTING PART 4.6 XAI FUNCTION
# ============================================================

def call_part46_xai(
    xai_function,
    model,
    canonical,
    point,
    class_id,
):
    """
    Call the proven Part 4.6 XAI implementation.

    IMPORTANT:
        target_layer must be the actual PyTorch module,
        not the string layer name.
    """

    # --------------------------------------------------------
    # Resolve the actual target layer
    # --------------------------------------------------------

    target_layer = model

    for name in TARGET_LAYER_NAME.split("."):

        if not hasattr(
            target_layer,
            name,
        ):
            raise AttributeError(
                "Could not resolve target layer "
                f"'{TARGET_LAYER_NAME}'. "
                f"Missing component '{name}'."
            )

        target_layer = getattr(
            target_layer,
            name,
        )

    # --------------------------------------------------------
    # Verify it is a PyTorch module
    # --------------------------------------------------------

    if not isinstance(
        target_layer,
        torch.nn.Module,
    ):
        raise TypeError(
            "Resolved target layer is not a "
            "torch.nn.Module: "
            f"{type(target_layer)}"
        )

    print(
        f"Resolved target layer: "
        f"{TARGET_LAYER_NAME}"
    )

    print(
        f"Target layer type: "
        f"{type(target_layer).__name__}"
    )

    # --------------------------------------------------------
    # Call the exact Part 4.6 function
    # --------------------------------------------------------

    # Convert the canonical NumPy volume to the tensor
    # expected by the validated Part 4.6 XAI function.
    # --------------------------------------------------------
# Convert canonical volume to the exact tensor shape
# expected by the validated Part 4.6 implementation.
#
# NumPy canonical:
#     (D, H, W)
#
# Part 4.6 expects:
#     (C, D, H, W)
#
# Its own function then adds the batch dimension:
#     (1, C, D, H, W)
# --------------------------------------------------------

    image_tensor = torch.from_numpy(
        np.asarray(
        canonical,
        dtype=np.float32,
    )
    ).float()

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    print(
    f"Part 4.6 XAI input shape: "
    f"{tuple(image_tensor.shape)}"
)

    return xai_function(
    model=model,
    image=image_tensor,
    point=point,
    target_layer=target_layer,
)


# ============================================================
# 9. EXTRACT XAI ARRAY
# ============================================================

def extract_xai_array(result):

    if isinstance(result, dict):

        possible_keys = [
            "xai",
            "cam",
            "attribution",
            "attributions",
        ]

        for key in possible_keys:

            if key in result:

                value = np.asarray(
                    result[key]
                )

                if value.ndim == 3:
                    return value.astype(
                        np.float32
                    )

    if isinstance(result, np.ndarray):

        if result.ndim == 3:
            return result.astype(
                np.float32
            )

    raise RuntimeError(
        "Could not extract a 3D XAI array "
        "from Part 4.6 output."
    )


# ============================================================
# 10. LOCALIZATION METRICS
# ============================================================

def evaluate_localization(
    xai,
    point,
):
    """
    Evaluate attribution concentration relative
    to the actual annotation point.
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

    xai = np.maximum(
        xai,
        0.0,
    )

    total = float(
        xai.sum()
    )

    if total <= 0:
        return {
            "centroid_z": None,
            "centroid_y": None,
            "centroid_x": None,
            "centroid_distance": None,
            "max_distance": None,
            "r2_concentration": 0.0,
            "r4_concentration": 0.0,
            "r6_concentration": 0.0,
            "r10_concentration": 0.0,
        }

    z_grid, y_grid, x_grid = np.indices(
        xai.shape,
        dtype=np.float32,
    )

    centroid_z = float(
        (z_grid * xai).sum()
        / total
    )

    centroid_y = float(
        (y_grid * xai).sum()
        / total
    )

    centroid_x = float(
        (x_grid * xai).sum()
        / total
    )

    target_z = float(
        point["z"]
    )

    target_y = float(
        point["y"]
    )

    target_x = float(
        point["x"]
    )

    centroid_distance = float(
        np.sqrt(
            (
                centroid_z
                - target_z
            ) ** 2
            +
            (
                centroid_y
                - target_y
            ) ** 2
            +
            (
                centroid_x
                - target_x
            ) ** 2
        )
    )

    max_index = np.unravel_index(
        np.argmax(xai),
        xai.shape,
    )

    max_distance = float(
        np.sqrt(
            (
                max_index[0]
                - target_z
            ) ** 2
            +
            (
                max_index[1]
                - target_y
            ) ** 2
            +
            (
                max_index[2]
                - target_x
            ) ** 2
        )
    )

    distance = np.sqrt(
        (
            z_grid
            - target_z
        ) ** 2
        +
        (
            y_grid
            - target_y
        ) ** 2
        +
        (
            x_grid
            - target_x
        ) ** 2
    )

    def concentration(radius):

        mask = (
            distance <= radius
        )

        return float(
            xai[mask].sum()
            / total
        )

    return {
        "centroid_z": centroid_z,
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_distance": centroid_distance,
        "max_distance": max_distance,
        "r2_concentration": concentration(2),
        "r4_concentration": concentration(4),
        "r6_concentration": concentration(6),
        "r10_concentration": concentration(10),
    }


# ============================================================
# 11. VISUALIZATION
# ============================================================

def save_xai_visualization(
    xai,
    point,
    disease,
    level,
    output_path,
):
    """
    Save axial, coronal and sagittal XAI views
    with the actual RSNA annotation point.
    """

    xai = np.asarray(
        xai,
        dtype=np.float32,
    )

    xai = np.maximum(
        xai,
        0.0,
    )

    if xai.max() > 0:
        xai_display = (
            xai
            / xai.max()
        )
    else:
        xai_display = xai

    z = int(
        np.clip(
            round(point["z"]),
            0,
            xai.shape[0] - 1,
        )
    )

    y = int(
        np.clip(
            round(point["y"]),
            0,
            xai.shape[1] - 1,
        )
    )

    x = int(
        np.clip(
            round(point["x"]),
            0,
            xai.shape[2] - 1,
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    # --------------------------------------------------------
    # Axial
    # --------------------------------------------------------

    axes[0].imshow(
        xai_display[z],
        cmap="hot",
        origin="lower",
    )

    axes[0].scatter(
        x,
        y,
        marker="x",
        s=100,
        linewidths=2,
    )

    axes[0].set_title(
        f"Axial Z={z}"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # Coronal
    # --------------------------------------------------------

    axes[1].imshow(
        xai_display[:, y, :],
        cmap="hot",
        origin="lower",
        aspect="auto",
    )

    axes[1].scatter(
        x,
        z,
        marker="x",
        s=100,
        linewidths=2,
    )

    axes[1].set_title(
        f"Coronal Y={y}"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # Sagittal
    # --------------------------------------------------------

    axes[2].imshow(
        xai_display[:, :, x],
        cmap="hot",
        origin="lower",
        aspect="auto",
    )

    axes[2].scatter(
        y,
        z,
        marker="x",
        s=100,
        linewidths=2,
    )

    axes[2].set_title(
        f"Sagittal X={x}"
    )

    axes[2].axis("off")

    fig.suptitle(
        f"{disease} | {level} | "
        "Point-Targeted 3D XAI",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. SAVE RAW XAI
# ============================================================

def save_xai_array(
    xai,
    disease,
    level,
):

    disease_dir = (
        OUTPUT_DIR
        / normalize_level(
            disease
        )
    )

    disease_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{normalize_level(level)}_xai.npy"
    )

    path = (
        disease_dir
        / filename
    )

    np.save(
        path,
        xai.astype(
            np.float32
        ),
    )

    return path


# ============================================================
# 13. MODEL ACCESS
# ============================================================

def obtain_model():

    """
    Reuse the model that Part 4.6 already knows how to load.

    The function searches the Part 4.6 module for a suitable
    model-loading helper.
    """

    candidates = [
        "load_model",
        "build_model",
        "create_model",
        "get_model",
    ]

    for name in candidates:

        function = getattr(
            part46,
            name,
            None,
        )

        if not callable(function):
            continue

        try:

            signature = inspect.signature(
                function
            )

            kwargs = {}

            if "device" in signature.parameters:
                kwargs["device"] = DEVICE

            if "checkpoint_path" in signature.parameters:

                checkpoint = getattr(
                    part46,
                    "CHECKPOINT_PATH",
                    None,
                )

                if checkpoint is None:
                    checkpoint = getattr(
                        part46,
                        "CHECKPOINT",
                        None,
                    )

                if checkpoint is not None:
                    kwargs[
                        "checkpoint_path"
                    ] = checkpoint

            model = function(
                **kwargs
            )

            if model is not None:

                print(
                    f"Model loaded using "
                    f"Part 4.6 helper: {name}"
                )

                return model

        except TypeError:
            continue

    raise RuntimeError(
        "\nCould not automatically locate "
        "the model-loading helper in Part 4.6.\n\n"
        "This is intentional rather than guessing "
        "the architecture/checkpoint loading code.\n"
        "If this occurs, send me the Part 4.6 "
        "script and I will connect the exact "
        "working loader."
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "PART 4.6B — MULTI-POINT "
        "POINT-TARGETED 3D XAI"
    )
    print("=" * 78)

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Study: {STUDY_ID}"
    )

    print(
        f"Series: {SERIES_ID}"
    )

    print(
        f"Target layer: "
        f"{TARGET_LAYER_NAME}"
    )

    # --------------------------------------------------------
    # Load canonical case
    # --------------------------------------------------------

    canonical, points, geometry = (
        load_case_from_part220b()
    )

    points_table = print_points(
        points
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model = obtain_model()

    model.eval()

    # --------------------------------------------------------
    # XAI function
    # --------------------------------------------------------

    xai_function = (
        get_xai_function()
    )

    # --------------------------------------------------------
    # Process every point
    # --------------------------------------------------------

    results = []

    disease_counts = {
        2: 0,
        3: 0,
    }

    for point_index, point in enumerate(
        points,
        start=1,
    ):

        class_id = safe_int(
            point.get("class_id")
        )

        if class_id not in TARGET_DISEASES:
            continue

        disease = TARGET_DISEASES[
            class_id
        ]

        level = str(
            point.get(
                "level",
                "unknown",
            )
        )

        disease_counts[
            class_id
        ] += 1

        print("\n")
        print("=" * 78)

        print(
            f"POINT {point_index}"
        )

        print(
            f"Disease: {disease}"
        )

        print(
            f"Level: {level}"
        )

        print(
            f"Class: {class_id}"
        )

        print(
            "Canonical point:"
        )

        print(
            f"    z = {point['z']:.6f}"
        )

        print(
            f"    y = {point['y']:.6f}"
        )

        print(
            f"    x = {point['x']:.6f}"
        )

        print("=" * 78)

        try:

            result = call_part46_xai(
                xai_function=xai_function,
                model=model,
                canonical=canonical,
                point=point,
                class_id=class_id,
            )

            xai = extract_xai_array(
                result
            )

            metrics = (
                evaluate_localization(
                    xai=xai,
                    point=point,
                )
            )

            # ------------------------------------------------
            # Target probability/logit if returned
            # ------------------------------------------------

            target_logit = None

            if isinstance(
                result,
                dict,
            ):

                if (
                    "target_logit"
                    in result
                ):

                    target_logit = safe_float(
                        result[
                            "target_logit"
                        ]
                    )

            target_probability = None

            if target_logit is not None:

                target_probability = float(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -target_logit
                        )
                    )
                )

            # ------------------------------------------------
            # Save XAI
            # ------------------------------------------------

            xai_path = save_xai_array(
                xai=xai,
                disease=disease,
                level=level,
            )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            disease_dir = (
                OUTPUT_DIR
                / normalize_level(
                    disease
                )
            )

            image_path = (
                disease_dir
                / f"{normalize_level(level)}.png"
            )

            save_xai_visualization(
                xai=xai,
                point=point,
                disease=disease,
                level=level,
                output_path=image_path,
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            row = {
                "point_index": point_index,
                "study_id": STUDY_ID,
                "series_id": SERIES_ID,
                "class_id": class_id,
                "disease": disease,
                "level": level,
                "point_z": float(
                    point["z"]
                ),
                "point_y": float(
                    point["y"]
                ),
                "point_x": float(
                    point["x"]
                ),
                "target_logit": target_logit,
                "target_probability": (
                    target_probability
                ),
                "xai_shape": str(
                    tuple(xai.shape)
                ),
                **metrics,
                "xai_file": str(
                    xai_path
                ),
                "visualization_file": str(
                    image_path
                ),
            }

            results.append(
                row
            )

            print(
                "\nLocalization metrics:"
            )

            print(
                f"Centroid distance: "
                f"{metrics['centroid_distance']}"
            )

            print(
                f"Maximum attribution distance: "
                f"{metrics['max_distance']}"
            )

            print(
                f"R2: "
                f"{metrics['r2_concentration']:.6f}"
            )

            print(
                f"R4: "
                f"{metrics['r4_concentration']:.6f}"
            )

            print(
                f"R6: "
                f"{metrics['r6_concentration']:.6f}"
            )

            print(
                f"R10: "
                f"{metrics['r10_concentration']:.6f}"
            )

        except Exception as exc:

            print(
                "\nERROR processing "
                f"{disease} {level}:"
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "point_index": point_index,
                    "study_id": STUDY_ID,
                    "series_id": SERIES_ID,
                    "class_id": class_id,
                    "disease": disease,
                    "level": level,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    # ========================================================
    # 15. SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        / "point_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Disease-level summary
    # --------------------------------------------------------

    summary = {
        "part": "4.6B",
        "study_id": STUDY_ID,
        "series_id": SERIES_ID,
        "target_layer": TARGET_LAYER_NAME,
        "model_shape": list(
            canonical.shape
        ),
        "device": str(DEVICE),
        "total_points_available": int(
            len(points)
        ),
        "target_points": int(
            len(results)
        ),
        "disease_counts": {
            "LFNN": int(
                disease_counts[2]
            ),
            "RFNN": int(
                disease_counts[3]
            ),
        },
        "successful_points": int(
            results_df[
                "error"
            ].isna().sum()
        )
        if "error" in results_df.columns
        else int(len(results_df)),
    }

    # --------------------------------------------------------
    # Aggregate disease statistics
    # --------------------------------------------------------

    disease_summary = {}

    if not results_df.empty:

        valid = results_df[
            ~results_df.columns.isin(
                ["error"]
            ).any()
        ] if False else results_df

        for class_id, disease in (
            TARGET_DISEASES.items()
        ):

            subset = results_df[
                results_df[
                    "class_id"
                ]
                == class_id
            ].copy()

            if subset.empty:
                continue

            numeric_columns = [
                "centroid_distance",
                "max_distance",
                "r2_concentration",
                "r4_concentration",
                "r6_concentration",
                "r10_concentration",
            ]

            disease_summary[
                disease
            ] = {}

            for column in numeric_columns:

                if column in subset.columns:

                    values = pd.to_numeric(
                        subset[column],
                        errors="coerce",
                    ).dropna()

                    if len(values):

                        disease_summary[
                            disease
                        ][column] = {
                            "mean": float(
                                values.mean()
                            ),
                            "median": float(
                                values.median()
                            ),
                            "std": float(
                                values.std(
                                    ddof=0
                                )
                            ),
                            "min": float(
                                values.min()
                            ),
                            "max": float(
                                values.max()
                            ),
                        }

    summary[
        "disease_summary"
    ] = disease_summary

    json_path = (
        OUTPUT_DIR
        / "part46b_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # 16. FINAL REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "part46b_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PART 4.6B\n"
        )

        f.write(
            "MULTI-POINT / MULTI-LEVEL "
            "POINT-TARGETED 3D XAI VALIDATION\n"
        )

        f.write(
            "=" * 72
            + "\n\n"
        )

        f.write(
            f"Study: {STUDY_ID}\n"
        )

        f.write(
            f"Series: {SERIES_ID}\n"
        )

        f.write(
            f"Target layer: "
            f"{TARGET_LAYER_NAME}\n"
        )

        f.write(
            f"Canonical shape: "
            f"{canonical.shape}\n"
        )

        f.write(
            f"Available points: "
            f"{len(points)}\n"
        )

        f.write(
            f"Processed results: "
            f"{len(results)}\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "- No model retraining.\n"
        )

        f.write(
            "- No checkpoint modification.\n"
        )

        f.write(
            "- No Part 2.20B geometry modification.\n"
        )

        f.write(
            "- No dashboard modification.\n"
        )

        f.write(
            "- No voxel-wise ground truth fabricated.\n"
        )

        f.write(
            "- RSNA annotation points are used "
            "as point targets only.\n"
        )

    # ========================================================
    # 17. FINAL CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 78)
    print(
        "PART 4.6B COMPLETE"
    )
    print("=" * 78)

    print(
        f"\nResults CSV:\n{csv_path}"
    )

    print(
        f"\nSummary JSON:\n{json_path}"
    )

    print(
        f"\nReport:\n{report_path}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nProcessed disease points:"
    )

    for class_id, disease in (
        TARGET_DISEASES.items()
    ):

        count = sum(
            1
            for row in results
            if row.get(
                "class_id"
            )
            == class_id
            and "error"
            not in row
        )

        print(
            f"  {disease}: {count}"
        )

    print(
        "\nNo training was performed."
    )

    print(
        "No checkpoint was modified."
    )

    print(
        "No geometry pipeline was modified."
    )

    print(
        "=" * 78
    )


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()