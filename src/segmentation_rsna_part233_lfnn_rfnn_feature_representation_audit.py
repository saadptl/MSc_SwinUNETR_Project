"""
PART 2.33
LFNN/RFNN FEATURE REPRESENTATION AUDIT

Purpose
-------
Inspect the learned SwinUNETR representation around paired:

    LFNN = class 2
    RFNN = class 3

Pairs are:
    same study
    same series
    same spinal level

This is an analysis-only experiment.

NO:
    - training
    - checkpoint modification
    - dashboard modification
    - voxel ground-truth creation

We compare:
    1. intermediate feature vectors
    2. feature cosine similarity
    3. feature L2 distance
    4. feature magnitude
    5. local feature statistics
    6. final logits
    7. LFNN/RFNN logit margins
    8. background-vs-foreground response

Important:
SwinUNETR internals can differ between MONAI versions.
The script therefore discovers candidate 3D feature tensors
during a forward pass rather than assuming a particular
private layer name.
"""

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src import (
    segmentation_rsna_part220b_geometry_corrected_training as p220b
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part233_lfnn_rfnn_feature_representation_audit"
)

TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONSTANTS
# ============================================================

LFNN = 2
RFNN = 3

CLASS_NAMES = p220b.CLASS_NAMES

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def load_checkpoint(model):

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=p220b.DEVICE,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        f"Checkpoint missing keys: "
        f"{len(missing)}"
    )

    print(
        f"Checkpoint unexpected keys: "
        f"{len(unexpected)}"
    )


# ============================================================
# VALIDATION MANIFEST
# ============================================================

def build_validation_manifest(
    manifest,
):

    validation = (
        p220b.select_validation_series(
            manifest
        )
    )

    if isinstance(
        validation,
        pd.DataFrame,
    ):

        ids = (
            validation[
                [
                    "study_id",
                    "series_id",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

    else:

        rows = []

        for item in validation:

            if isinstance(
                item,
                dict,
            ):

                rows.append(
                    (
                        str(
                            item["study_id"]
                        ),
                        str(
                            item["series_id"]
                        ),
                    )
                )

            elif (
                isinstance(
                    item,
                    (
                        tuple,
                        list,
                    ),
                )
                and len(item) >= 2
            ):

                rows.append(
                    (
                        str(item[0]),
                        str(item[1]),
                    )
                )

        ids = pd.DataFrame(
            rows,
            columns=[
                "study_id",
                "series_id",
            ],
        )

    m = manifest.copy()

    for column in [
        "study_id",
        "series_id",
    ]:

        m[column] = (
            m[column]
            .astype(str)
        )

        ids[column] = (
            ids[column]
            .astype(str)
        )

    result = m.merge(
        ids[
            [
                "study_id",
                "series_id",
            ]
        ].drop_duplicates(),
        on=[
            "study_id",
            "series_id",
        ],
        how="inner",
    )

    return result


# ============================================================
# MODEL OUTPUT HANDLING
# ============================================================

def unwrap_model_output(
    output,
):

    if isinstance(
        output,
        (tuple, list),
    ):

        if len(output) == 0:
            raise RuntimeError(
                "Model returned an empty tuple/list."
            )

        return unwrap_model_output(
            output[0]
        )

    if isinstance(
        output,
        dict,
    ):

        # Try common output names.
        for key in [
            "logits",
            "out",
            "output",
            "pred",
        ]:

            if key in output:

                return unwrap_model_output(
                    output[key]
                )

        raise RuntimeError(
            "Could not find tensor in model dict output."
        )

    return output


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureCapture:

    def __init__(self):

        self.records = []

        self.handles = []

    def clear(self):

        self.records.clear()

    def close(self):

        for handle in self.handles:

            handle.remove()

        self.handles.clear()


def discover_feature_modules(
    model,
):

    """
    Register hooks on leaf modules.

    We only keep outputs that are tensors with:

        batch dimension
        channel dimension
        >= 3 spatial dimensions

    This lets the script work without assuming
    exact MONAI SwinUNETR private attribute names.
    """

    candidates = []

    for name, module in model.named_modules():

        children = list(
            module.children()
        )

        if children:
            continue

        def make_hook(
            module_name
        ):

            def hook(
                module,
                inputs,
                output,
            ):

                tensor = None

                if torch.is_tensor(
                    output
                ):

                    tensor = output

                elif isinstance(
                    output,
                    (tuple, list),
                ):

                    for item in output:

                        if torch.is_tensor(
                            item
                        ):

                            tensor = item

                            break

                if tensor is None:
                    return

                if tensor.ndim < 4:
                    return

                if tensor.shape[0] != 1:
                    return

                self_capture.records.append(
                    {
                        "name":
                            module_name,
                        "shape":
                            tuple(
                                tensor.shape
                            ),
                        "tensor":
                            tensor,
                    }
                )

            return hook

        self_capture = FeatureCapture()

        handle = module.register_forward_hook(
            make_hook(name)
        )

        self_capture.handles.append(
            handle
        )

        candidates.append(
            self_capture
        )

    return candidates


# ============================================================
# SIMPLER GLOBAL FEATURE CAPTURE
# ============================================================

def capture_all_features(
    model,
    x,
):

    captured = {}

    handles = []

    for name, module in model.named_modules():

        if list(
            module.children()
        ):
            continue

        def make_hook(
            module_name
        ):

            def hook(
                module,
                inputs,
                output,
            ):

                tensor = None

                if torch.is_tensor(
                    output
                ):

                    tensor = output

                elif isinstance(
                    output,
                    (tuple, list),
                ):

                    for item in output:

                        if torch.is_tensor(
                            item
                        ):

                            tensor = item
                            break

                if tensor is None:
                    return

                if tensor.ndim >= 4:

                    captured[
                        module_name
                    ] = tensor.detach()

            return hook

        handles.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )

    with torch.no_grad():

        output = model(x)

    for handle in handles:

        handle.remove()

    return output, captured


# ============================================================
# SPATIAL FEATURE SAMPLING
# ============================================================

def feature_vector_at_point(
    feature,
    point,
    image_shape,
):

    """
    feature:
        [1,C,D,H,W]

    point:
        canonical model coordinates.

    We map the point from image resolution
    to the feature-map resolution.
    """

    if feature.ndim != 5:
        return None

    _, channels, fd, fh, fw = (
        feature.shape
    )

    D, H, W = image_shape

    z = float(point["z"])
    y = float(point["y"])
    x = float(point["x"])

    fz = (
        z
        / max(
            D - 1,
            1,
        )
        * max(
            fd - 1,
            1,
        )
    )

    fy = (
        y
        / max(
            H - 1,
            1,
        )
        * max(
            fh - 1,
            1,
        )
    )

    fx = (
        x
        / max(
            W - 1,
            1,
        )
        * max(
            fw - 1,
            1,
        )
    )

    zi = int(
        np.clip(
            round(fz),
            0,
            fd - 1,
        )
    )

    yi = int(
        np.clip(
            round(fy),
            0,
            fh - 1,
        )
    )

    xi = int(
        np.clip(
            round(fx),
            0,
            fw - 1,
        )
    )

    vector = (
        feature[
            0,
            :,
            zi,
            yi,
            xi,
        ]
        .float()
        .cpu()
    )

    return vector


# ============================================================
# FEATURE SUMMARY
# ============================================================

def summarize_feature_pair(
    left_vector,
    right_vector,
):

    if (
        left_vector is None
        or right_vector is None
    ):

        return {
            "feature_cosine":
                np.nan,
            "feature_l2":
                np.nan,
            "lfnn_feature_norm":
                np.nan,
            "rfnn_feature_norm":
                np.nan,
            "feature_abs_difference_mean":
                np.nan,
        }

    left = (
        left_vector
        .reshape(1, -1)
    )

    right = (
        right_vector
        .reshape(1, -1)
    )

    cosine = F.cosine_similarity(
        left,
        right,
        dim=1,
    ).item()

    l2 = torch.norm(
        left_vector
        - right_vector
    ).item()

    left_norm = (
        torch.norm(
            left_vector
        ).item()
    )

    right_norm = (
        torch.norm(
            right_vector
        ).item()
    )

    abs_difference = (
        torch.mean(
            torch.abs(
                left_vector
                - right_vector
            )
        ).item()
    )

    return {
        "feature_cosine":
            float(cosine),
        "feature_l2":
            float(l2),
        "lfnn_feature_norm":
            float(left_norm),
        "rfnn_feature_norm":
            float(right_norm),
        "feature_abs_difference_mean":
            float(abs_difference),
    }


# ============================================================
# LOGIT EXTRACTION
# ============================================================

def point_logits(
    logits,
    point,
):

    if logits.ndim != 5:
        raise RuntimeError(
            f"Expected 5D logits, got {logits.shape}"
        )

    _, classes, D, H, W = (
        logits.shape
    )

    z = int(
        np.clip(
            round(
                float(
                    point["z"]
                )
            ),
            0,
            D - 1,
        )
    )

    y = int(
        np.clip(
            round(
                float(
                    point["y"]
                )
            ),
            0,
            H - 1,
        )
    )

    x = int(
        np.clip(
            round(
                float(
                    point["x"]
                )
            ),
            0,
            W - 1,
        )
    )

    values = (
        logits[
            0,
            :,
            z,
            y,
            x,
        ]
        .float()
        .cpu()
    )

    return values


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PART 2.33 — LFNN/RFNN "
        "FEATURE REPRESENTATION AUDIT"
    )
    print("=" * 72)

    print(
        "GPU:",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = (
        p220b.load_manifest()
    )

    validation_manifest = (
        build_validation_manifest(
            manifest
        )
    )

    validation_pairs = (
        validation_manifest[
            [
                "study_id",
                "series_id",
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"Validation cases: "
        f"{len(validation_pairs)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        p220b.build_model()
    )

    load_checkpoint(
        model
    )

    model.to(
        p220b.DEVICE
    )

    model.eval()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    records = []

    # Feature shapes encountered
    feature_shape_counts = {}

    processed = 0

    # --------------------------------------------------------
    # Cases
    # --------------------------------------------------------

    for (
        study_id,
        series_id,
    ), group in validation_manifest.groupby(
        [
            "study_id",
            "series_id",
        ],
        sort=True,
    ):

        image, points, geometry = (
            p220b.load_case(
                str(study_id),
                str(series_id),
                group.copy(),
            )
        )

        x = torch.from_numpy(
            image.astype(
                np.float32
            )
        )

        x = (
            x
            .unsqueeze(0)
            .unsqueeze(0)
            .to(
                p220b.DEVICE
            )
        )

        # ----------------------------------------------------
        # Forward + feature capture
        # ----------------------------------------------------

        with torch.no_grad():

            raw_output, captured = (
                capture_all_features(
                    model,
                    x,
                )
            )

        logits = unwrap_model_output(
            raw_output
        )

        # ----------------------------------------------------
        # Select useful 3D feature tensors
        # ----------------------------------------------------

        feature_candidates = []

        for name, tensor in captured.items():

            if tensor.ndim != 5:
                continue

            if tensor.shape[0] != 1:
                continue

            _, channels, fd, fh, fw = (
                tensor.shape
            )

            # Ignore the final six-channel segmentation
            # output if it was captured.
            if channels == 6:
                continue

            # Need actual spatial dimensions.
            if min(
                fd,
                fh,
                fw,
            ) <= 1:
                continue

            feature_candidates.append(
                (
                    name,
                    tensor,
                )
            )

            key = (
                name,
                tuple(
                    tensor.shape
                ),
            )

            feature_shape_counts[key] = (
                feature_shape_counts.get(
                    key,
                    0
                )
                + 1
            )

        if not feature_candidates:

            raise RuntimeError(
                "No suitable intermediate "
                "3D feature tensors found."
            )

        # ----------------------------------------------------
        # Choose the deepest/highest-channel useful feature
        #
        # Prefer:
        # 1. highest channel count
        # 2. reasonable spatial map
        # ----------------------------------------------------

        feature_candidates.sort(
            key=lambda item: (
                item[1].shape[1],
                item[1].numel(),
            )
        )

        feature_name, feature = (
            feature_candidates[-1]
        )

        print(
            f"Case {processed + 1:02d}/"
            f"{len(validation_pairs):02d}"
            f" | feature={feature_name}"
            f" | shape={tuple(feature.shape)}"
        )

        # ----------------------------------------------------
        # Build point lookup
        # ----------------------------------------------------

        lookup = {}

        for point in points:

            class_id = int(
                point["class_id"]
            )

            if class_id not in {
                LFNN,
                RFNN,
            }:
                continue

            key = (
                str(
                    point["level"]
                ),
                class_id,
            )

            lookup[key] = point

        # ----------------------------------------------------
        # Pair LFNN/RFNN
        # ----------------------------------------------------

        levels = sorted(
            {
                level
                for level, class_id
                in lookup.keys()
            }
        )

        for level in levels:

            left = lookup.get(
                (
                    level,
                    LFNN,
                )
            )

            right = lookup.get(
                (
                    level,
                    RFNN,
                )
            )

            if (
                left is None
                or right is None
            ):
                continue

            # ----------------------------------------------
            # Feature vectors
            # ----------------------------------------------

            left_vector = (
                feature_vector_at_point(
                    feature,
                    left,
                    image.shape,
                )
            )

            right_vector = (
                feature_vector_at_point(
                    feature,
                    right,
                    image.shape,
                )
            )

            feature_summary = (
                summarize_feature_pair(
                    left_vector,
                    right_vector,
                )
            )

            # ----------------------------------------------
            # Final logits
            # ----------------------------------------------

            left_logits = (
                point_logits(
                    logits,
                    left,
                )
            )

            right_logits = (
                point_logits(
                    logits,
                    right,
                )
            )

            left_prob = torch.softmax(
                left_logits,
                dim=0,
            )

            right_prob = torch.softmax(
                right_logits,
                dim=0,
            )

            # ----------------------------------------------
            # Margins
            # ----------------------------------------------

            left_lfnn_margin = (
                left_logits[LFNN]
                - left_logits[0]
            ).item()

            left_rfnn_margin = (
                left_logits[RFNN]
                - left_logits[0]
            ).item()

            right_lfnn_margin = (
                right_logits[LFNN]
                - right_logits[0]
            ).item()

            right_rfnn_margin = (
                right_logits[RFNN]
                - right_logits[0]
            ).item()

            record = {

                "study_id":
                    str(study_id),

                "series_id":
                    str(series_id),

                "level":
                    str(level),

                "feature_name":
                    feature_name,

                "feature_channels":
                    int(
                        feature.shape[1]
                    ),

                "feature_depth":
                    int(
                        feature.shape[2]
                    ),

                "feature_height":
                    int(
                        feature.shape[3]
                    ),

                "feature_width":
                    int(
                        feature.shape[4]
                    ),

                # ------------------------------------------
                # Feature comparison
                # ------------------------------------------

                **feature_summary,

                # ------------------------------------------
                # LFNN logits
                # ------------------------------------------

                "lfnn_logit_background":
                    float(
                        left_logits[0]
                    ),

                "lfnn_logit_lfnn":
                    float(
                        left_logits[LFNN]
                    ),

                "lfnn_logit_rfnn":
                    float(
                        left_logits[RFNN]
                    ),

                "lfnn_prob_background":
                    float(
                        left_prob[0]
                    ),

                "lfnn_prob_lfnn":
                    float(
                        left_prob[LFNN]
                    ),

                "lfnn_prob_rfnn":
                    float(
                        left_prob[RFNN]
                    ),

                "lfnn_lfnn_margin":
                    float(
                        left_lfnn_margin
                    ),

                "lfnn_rfnn_margin":
                    float(
                        left_rfnn_margin
                    ),

                # ------------------------------------------
                # RFNN logits
                # ------------------------------------------

                "rfnn_logit_background":
                    float(
                        right_logits[0]
                    ),

                "rfnn_logit_lfnn":
                    float(
                        right_logits[LFNN]
                    ),

                "rfnn_logit_rfnn":
                    float(
                        right_logits[RFNN]
                    ),

                "rfnn_prob_background":
                    float(
                        right_prob[0]
                    ),

                "rfnn_prob_lfnn":
                    float(
                        right_prob[LFNN]
                    ),

                "rfnn_prob_rfnn":
                    float(
                        right_prob[RFNN]
                    ),

                "rfnn_lfnn_margin":
                    float(
                        right_lfnn_margin
                    ),

                "rfnn_rfnn_margin":
                    float(
                        right_rfnn_margin
                    ),

                # ------------------------------------------
                # Paired differences
                # ------------------------------------------

                "rfnn_minus_lfnn_background_prob":
                    float(
                        right_prob[0]
                        - left_prob[0]
                    ),

                "rfnn_minus_lfnn_lfnn_prob":
                    float(
                        right_prob[LFNN]
                        - left_prob[LFNN]
                    ),

                "rfnn_minus_lfnn_rfnn_prob":
                    float(
                        right_prob[RFNN]
                        - left_prob[RFNN]
                    ),

                "rfnn_minus_lfnn_rfnn_margin":
                    float(
                        right_rfnn_margin
                        - left_rfnn_margin
                    ),

                "rfnn_minus_lfnn_lfnn_margin":
                    float(
                        right_lfnn_margin
                        - left_lfnn_margin
                    ),
            }

            records.append(
                record
            )

        processed += 1

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        records
    )

    if df.empty:

        raise RuntimeError(
            "No LFNN/RFNN feature pairs generated."
        )

    df.to_csv(
        TABLE_DIR
        / "part233_feature_pairs.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "validation_cases":
            int(processed),

        "paired_observations":
            int(len(df)),

        "mean_feature_cosine":
            float(
                df[
                    "feature_cosine"
                ].mean()
            ),

        "median_feature_cosine":
            float(
                df[
                    "feature_cosine"
                ].median()
            ),

        "mean_feature_l2":
            float(
                df[
                    "feature_l2"
                ].mean()
            ),

        "mean_feature_abs_difference":
            float(
                df[
                    "feature_abs_difference_mean"
                ].mean()
            ),

        "lfnn_mean_feature_norm":
            float(
                df[
                    "lfnn_feature_norm"
                ].mean()
            ),

        "rfnn_mean_feature_norm":
            float(
                df[
                    "rfnn_feature_norm"
                ].mean()
            ),

        "lfnn_mean_background_probability":
            float(
                df[
                    "lfnn_prob_background"
                ].mean()
            ),

        "rfnn_mean_background_probability":
            float(
                df[
                    "rfnn_prob_background"
                ].mean()
            ),

        "lfnn_mean_lfnn_probability":
            float(
                df[
                    "lfnn_prob_lfnn"
                ].mean()
            ),

        "rfnn_mean_lfnn_probability":
            float(
                df[
                    "rfnn_prob_lfnn"
                ].mean()
            ),

        "lfnn_mean_rfnn_probability":
            float(
                df[
                    "lfnn_prob_rfnn"
                ].mean()
            ),

        "rfnn_mean_rfnn_probability":
            float(
                df[
                    "rfnn_prob_rfnn"
                ].mean()
            ),

        "mean_rfnn_minus_lfnn_rfnn_probability":
            float(
                df[
                    "rfnn_minus_lfnn_rfnn_prob"
                ].mean()
            ),

        "mean_rfnn_minus_lfnn_background_probability":
            float(
                df[
                    "rfnn_minus_lfnn_background_prob"
                ].mean()
            ),

        "mean_rfnn_minus_lfnn_rfnn_margin":
            float(
                df[
                    "rfnn_minus_lfnn_rfnn_margin"
                ].mean()
            ),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        TABLE_DIR
        / "part233_summary.csv",
        index=False,
    )

    # ========================================================
    # FEATURE SHAPE REPORT
    # ========================================================

    feature_shapes = []

    for (
        key,
        count,
    ) in feature_shape_counts.items():

        name, shape = key

        feature_shapes.append(
            {
                "feature_name":
                    name,
                "shape":
                    str(shape),
                "cases":
                    count,
            }
        )

    feature_shape_df = pd.DataFrame(
        feature_shapes
    )

    feature_shape_df.to_csv(
        TABLE_DIR
        / "part233_feature_shapes.csv",
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "PART 2.33 — LFNN/RFNN "
        "FEATURE REPRESENTATION AUDIT"
    )

    report.append("")

    report.append(
        f"Validation cases: {processed}"
    )

    report.append(
        f"Paired LFNN/RFNN observations: {len(df)}"
    )

    report.append("")

    report.append(
        "SUMMARY:"
    )

    for key, value in summary.items():

        report.append(
            f"{key}: {value}"
        )

    report.append("")

    report.append(
        "FEATURE TENSORS DISCOVERED:"
    )

    report.append(
        feature_shape_df.to_string(
            index=False
        )
    )

    report.append("")

    report.append(
        "INTERPRETATION GUIDE:"
    )

    report.append(
        "High feature cosine similarity with "
        "very different final probabilities suggests "
        "the final classifier/decoder is producing "
        "the asymmetry."
    )

    report.append(
        "Low feature similarity between paired LFNN/RFNN "
        "points suggests the representation itself "
        "encodes substantially different responses."
    )

    report.append(
        "A strongly negative RFNN-minus-LFNN RFNN margin "
        "indicates suppression of the RFNN class at RFNN points."
    )

    report.append("")

    report.append(
        "No training was performed."
    )

    report.append(
        "No checkpoint was modified."
    )

    report.append(
        "No dashboard was modified."
    )

    report.append(
        "No voxel-wise ground truth was fabricated."
    )

    report.append("")

    report.append(
        "RSNA annotations remain point/localization "
        "annotations rather than manual voxel masks."
    )

    report_path = (
        REPORT_DIR
        / "part233_report.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PART 2.33 COMPLETE"
    )
    print("=" * 72)

    print(
        f"Validation cases: "
        f"{processed}"
    )

    print(
        f"Paired observations: "
        f"{len(df)}"
    )

    print(
        f"Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Training performed: NO"
    )

    print(
        "Checkpoint modified: NO"
    )

    print(
        "Dashboard modified: NO"
    )


if __name__ == "__main__":
    main()