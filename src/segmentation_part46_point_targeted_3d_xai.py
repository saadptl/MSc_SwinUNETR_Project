"""
PART 4.6
POINT-TARGETED 3D XAI VALIDATION

Purpose:
    Generate genuine point-targeted gradient attribution maps
    using the physical-space RSNA annotation points.

Important:
    - No training
    - No checkpoint modification
    - No fabricated voxel masks
    - Uses the existing Part 3.3 checkpoint
    - Uses the existing Part 2.20B physical-space pipeline
"""

from pathlib import Path
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# IMPORTANT:
# Keep the validated Part 2.20B geometry pipeline unchanged.
import segmentation_rsna_part220b_geometry_corrected_training as part220b


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

STUDY_ID = 7143189
SERIES_ID = 3219733239

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part46_point_targeted_3d_xai"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Existing Part 3.3 checkpoint.
CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_balanced_symmetry_refinement"
    / "checkpoints"
    / "part33_best_development_macro.pth"
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MODEL_SHAPE = (64, 96, 96)


# ---------------------------------------------------------
# DISEASE DEFINITIONS
# ---------------------------------------------------------

DISEASES = {
    1: "spinal_canal_stenosis",
    2: "left_neural_foraminal_narrowing",
    3: "right_neural_foraminal_narrowing",
    4: "left_subarticular_stenosis",
    5: "right_subarticular_stenosis",
}


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def build_model():
    """
    Build the exact model architecture used by Part 2.20B.
    """

    model = part220b.build_model()

    return model.to(DEVICE)


# ---------------------------------------------------------
# POINT NORMALIZATION
# ---------------------------------------------------------

def normalize_points(points):
    """
    Convert point records to a consistent list of dictionaries.
    """

    normalized = []

    for p in points:

        if isinstance(p, dict):

            q = dict(p)

        else:

            raise TypeError(
                f"Unexpected point type: {type(p)}"
            )

        if "class_id" not in q:
            continue

        normalized.append(q)

    return normalized


# ---------------------------------------------------------
# POINT-TARGETED XAI
# ---------------------------------------------------------

def compute_point_targeted_xai(
    model,
    image,
    point,
    target_layer,
):
    """
    Generate a Grad-CAM-style gradient attribution map for
    one disease class at one annotated model-grid point.

    IMPORTANT:
        The target is the disease logit at the actual
        annotation point.

        This is NOT the mean disease probability over
        the complete volume.

    The implementation deliberately avoids
    register_full_backward_hook(), because MONAI's
    decoder contains in-place residual operations that
    can conflict with PyTorch backward hooks.

    Instead, the gradient of the point-specific logit
    with respect to the captured target-layer activation
    is obtained explicitly with torch.autograd.grad().
    """

    activations = {}

    # -----------------------------------------------------
    # FORWARD HOOK
    # -----------------------------------------------------

    def forward_hook(module, inputs, output):

        activations["value"] = output


    h1 = target_layer.register_forward_hook(
        forward_hook
    )


    try:

        # -------------------------------------------------
        # RESET MODEL GRADIENTS
        # -------------------------------------------------

        model.zero_grad(set_to_none=True)


        # -------------------------------------------------
        # PREPARE INPUT
        # -------------------------------------------------

        x = image.to(DEVICE)

        if x.ndim == 4:
            x = x.unsqueeze(0)

        x = x.requires_grad_(True)


        # -------------------------------------------------
        # FORWARD PASS
        # -------------------------------------------------

        logits = model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]


        # -------------------------------------------------
        # VERIFY TARGET ACTIVATION
        # -------------------------------------------------

        if "value" not in activations:

            raise RuntimeError(
                "Target-layer activation was not captured."
            )


        activation = activations["value"]


        # -------------------------------------------------
        # EXTRACT MODEL-GRID POINT
        # -------------------------------------------------

        z = int(
            round(
                float(point["z"])
            )
        )

        y = int(
            round(
                float(point["y"])
            )
        )

        xcoord = int(
            round(
                float(point["x"])
            )
        )


        # -------------------------------------------------
        # BOUNDS CHECK
        # -------------------------------------------------

        original_point = (
            z,
            y,
            xcoord,
        )


        z = max(
            0,
            min(
                z,
                logits.shape[2] - 1
            )
        )

        y = max(
            0,
            min(
                y,
                logits.shape[3] - 1
            )
        )

        xcoord = max(
            0,
            min(
                xcoord,
                logits.shape[4] - 1
            )
        )


        if original_point != (
            z,
            y,
            xcoord,
        ):

            print(
                "WARNING: Point was clipped to "
                "model output bounds."
            )

            print(
                "Original point:",
                original_point
            )

            print(
                "Clipped point:",
                (
                    z,
                    y,
                    xcoord,
                )
            )


        # -------------------------------------------------
        # DISEASE CLASS
        # -------------------------------------------------

        class_id = int(
            point["class_id"]
        )


        # -------------------------------------------------
        # POINT-TARGETED DISEASE LOGIT
        # -------------------------------------------------

        target = logits[
            0,
            class_id,
            z,
            y,
            xcoord,
        ]


        # -------------------------------------------------
        # EXPLICIT AUTOGRAD
        # -------------------------------------------------

        gradient = torch.autograd.grad(
            outputs=target,
            inputs=activation,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]


        # -------------------------------------------------
        # GRAD-CAM CHANNEL WEIGHTS
        # -------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )


        # -------------------------------------------------
        # CHANNEL-WEIGHTED ACTIVATION
        # -------------------------------------------------

        cam = (
            weights * activation
        ).sum(
            dim=1
        )


        # -------------------------------------------------
        # POSITIVE ATTRIBUTION
        # -------------------------------------------------

        cam = F.relu(cam)


        # -------------------------------------------------
        # UPSAMPLE TO MODEL INPUT GRID
        # -------------------------------------------------

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=MODEL_SHAPE,
            mode="trilinear",
            align_corners=False,
        )


        # -------------------------------------------------
        # REMOVE BATCH / CHANNEL DIMENSIONS
        # -------------------------------------------------

        cam = cam[0, 0]


        # -------------------------------------------------
        # NORMALIZE
        # -------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()


        if (
            cam_max - cam_min
        ).abs() > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(
                cam
            )


        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        cam_np = (
            cam
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # -------------------------------------------------
        # RETURN STRUCTURE
        #
        # IMPORTANT:
        # main() expects:
        #     result["xai"]
        #     result["target_logit"]
        #     result["point_z"]
        #     result["point_y"]
        #     result["point_x"]
        # -------------------------------------------------

        return {

            "xai": cam_np,

            "target_logit": float(
                target
                .detach()
                .cpu()
                .item()
            ),

            "point_z": z,

            "point_y": y,

            "point_x": xcoord,

            "class_id": class_id,

            "activation_shape": tuple(
                activation.shape
            ),

            "gradient_shape": tuple(
                gradient.shape
            ),
        }


    finally:

        h1.remove()


# ---------------------------------------------------------
# DISTANCE / CONCENTRATION METRICS
# ---------------------------------------------------------

def evaluate_localization(
    xai,
    point,
    radii=(2, 4, 6, 10),
):
    """
    Evaluate how concentrated the attribution is around
    the annotation point.

    These are attribution-localization metrics only.
    They are NOT clinical localization metrics.
    """

    # -----------------------------------------------------
    # TARGET POINT
    # -----------------------------------------------------

    z0 = int(
        round(
            float(point["z"])
        )
    )

    y0 = int(
        round(
            float(point["y"])
        )
    )

    x0 = int(
        round(
            float(point["x"])
        )
    )

    # -----------------------------------------------------
    # CLIP POINT TO XAI VOLUME
    # -----------------------------------------------------

    z0 = max(
        0,
        min(
            z0,
            xai.shape[0] - 1,
        ),
    )

    y0 = max(
        0,
        min(
            y0,
            xai.shape[1] - 1,
        ),
    )

    x0 = max(
        0,
        min(
            x0,
            xai.shape[2] - 1,
        ),
    )

    # -----------------------------------------------------
    # 3D COORDINATE GRID
    # -----------------------------------------------------

    zz, yy, xx = np.indices(
        xai.shape
    )

    # -----------------------------------------------------
    # EUCLIDEAN DISTANCE FROM ANNOTATION POINT
    # -----------------------------------------------------

    distance = np.sqrt(
        (zz - z0) ** 2
        + (yy - y0) ** 2
        + (xx - x0) ** 2
    )

    result = {}

    # -----------------------------------------------------
    # TOTAL ATTRIBUTION
    # -----------------------------------------------------

    total = float(
        xai.sum()
    )

    # -----------------------------------------------------
    # ATTRIBUTION CENTROID
    # -----------------------------------------------------

    if total > 0:

        weighted_z = float(
            (xai * zz).sum()
            / total
        )

        weighted_y = float(
            (xai * yy).sum()
            / total
        )

        weighted_x = float(
            (xai * xx).sum()
            / total
        )

        centroid_distance = float(
            np.sqrt(
                (weighted_z - z0) ** 2
                + (weighted_y - y0) ** 2
                + (weighted_x - x0) ** 2
            )
        )

    else:

        centroid_distance = float(
            "inf"
        )

    result[
        "centroid_distance_voxels"
    ] = centroid_distance

    # -----------------------------------------------------
    # MAX-ATTRIBUTION DISTANCE
    # -----------------------------------------------------
    #
    # np.argmax() returns a flattened index.
    # Convert it back to (z, y, x) before indexing
    # the 3D distance array.
    # -----------------------------------------------------

    if xai.size > 0:

        max_index = np.unravel_index(
            np.argmax(xai),
            xai.shape,
        )

        result[
            "max_attribution_distance_voxels"
        ] = float(
            distance[max_index]
        )

    else:

        result[
            "max_attribution_distance_voxels"
        ] = float("inf")

    # -----------------------------------------------------
    # ATTRIBUTION CONCENTRATION
    # -----------------------------------------------------

    for radius in radii:

        mask = (
            distance <= radius
        )

        local_sum = float(
            xai[mask].sum()
        )

        if total > 0:

            concentration = (
                local_sum / total
            )

        else:

            concentration = 0.0

        result[
            f"attribution_concentration_r{radius}"
        ] = float(
            concentration
        )

    return result


# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

def save_visualization(
    image,
    probability,
    xai,
    point,
    disease_name,
    output_path,
):
    """
    Save sagittal, coronal and axial targeted-XAI views.
    """

    z = int(
        round(
            float(point["z"])
        )
    )

    y = int(
        round(
            float(point["y"])
        )
    )

    x = int(
        round(
            float(point["x"])
        )
    )


    # -----------------------------------------------------
    # ORIENTATION VIEWS
    # -----------------------------------------------------

    views = {

        "axial": (
            image[z, :, :],
            probability[z, :, :],
            xai[z, :, :],
        ),

        "coronal": (
            image[:, y, :],
            probability[:, y, :],
            xai[:, y, :],
        ),

        "sagittal": (
            image[:, :, x],
            probability[:, :, x],
            xai[:, :, x],
        ),
    }


    # -----------------------------------------------------
    # GENERATE FIGURES
    # -----------------------------------------------------

    for orientation, (
        mri,
        prob,
        attribution,
    ) in views.items():

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(18, 5),
        )


        # -------------------------------------------------
        # MRI
        # -------------------------------------------------

        axes[0].imshow(
            mri,
            cmap="gray",
            origin="lower",
        )

        axes[0].set_title(
            "MRI"
        )


        # -------------------------------------------------
        # DISEASE PROBABILITY
        # -------------------------------------------------

        axes[1].imshow(
            prob,
            cmap="magma",
            origin="lower",
            vmin=0,
            vmax=1,
        )

        axes[1].set_title(
            "Disease Probability"
        )


        # -------------------------------------------------
        # POINT-TARGETED XAI
        # -------------------------------------------------

        axes[2].imshow(
            attribution,
            cmap="viridis",
            origin="lower",
            vmin=0,
            vmax=1,
        )

        axes[2].set_title(
            "Point-Targeted 3D Attribution"
        )


        # -------------------------------------------------
        # MRI + XAI
        # -------------------------------------------------

        axes[3].imshow(
            mri,
            cmap="gray",
            origin="lower",
        )

        axes[3].imshow(
            attribution,
            cmap="jet",
            origin="lower",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )


        # -------------------------------------------------
        # POINT MARKER
        #
        # Coordinates correspond to the current 2D view.
        # -------------------------------------------------

        if orientation == "axial":

            marker_x = x
            marker_y = y

        elif orientation == "coronal":

            marker_x = x
            marker_y = z

        else:

            marker_x = y
            marker_y = z


        axes[3].scatter(
            [marker_x],
            [marker_y],
            marker="+",
            s=100,
        )


        axes[3].set_title(
            "MRI + Point-Targeted XAI"
        )


        # -------------------------------------------------
        # FIGURE TITLE
        # -------------------------------------------------

        fig.suptitle(
            f"Part 4.6 - {disease_name} - "
            f"{orientation.capitalize()}",
            fontsize=14,
        )


        for ax in axes:
            ax.axis("off")


        plt.tight_layout()


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        filename = (
            f"{disease_name}_{orientation}.png"
        )


        fig.savefig(
            output_path / filename,
            dpi=180,
            bbox_inches="tight",
        )


        plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "=" * 70
    )

    print(
        "PART 4.6"
    )

    print(
        "POINT-TARGETED 3D XAI VALIDATION"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # ENVIRONMENT
    # -----------------------------------------------------

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Study:",
        STUDY_ID,
    )

    print(
        "Series:",
        SERIES_ID,
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )


    # -----------------------------------------------------
    # MANIFEST
    # -----------------------------------------------------

    manifest = (
        part220b.load_manifest()
    )


    print(
        "Manifest rows:",
        len(manifest),
    )


    # -----------------------------------------------------
    # FILTER ONLY THIS STUDY + SERIES
    # -----------------------------------------------------

    case_manifest = manifest[
        (
            manifest["study_id"]
            .astype(str)
            == str(STUDY_ID)
        )
        &
        (
            manifest["series_id"]
            .astype(str)
            == str(SERIES_ID)
        )
    ].copy()


    print(
        "Case manifest rows:",
        len(case_manifest),
    )


    if case_manifest.empty:

        raise RuntimeError(
            f"No annotation points found for "
            f"Study {STUDY_ID}, "
            f"Series {SERIES_ID}"
        )


    # -----------------------------------------------------
    # LOAD CASE USING EXACT PART 2.20B PIPELINE
    # -----------------------------------------------------

    case_result = (
        part220b.load_case(
            STUDY_ID,
            SERIES_ID,
            case_manifest,
        )
    )


    if len(case_result) == 3:

        image, points, geometry = (
            case_result
        )

    else:

        image, points, geometry, *_ = (
            case_result
        )


    points = normalize_points(
        points
    )


    print(
        "Image shape:",
        np.asarray(image).shape,
    )


    print(
        "Points:",
        len(points),
    )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = build_model()


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )


    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint,
        ),
    )


    model.load_state_dict(
        state_dict,
        strict=True,
    )


    model.eval()


    # -----------------------------------------------------
    # TARGET LAYER
    # -----------------------------------------------------

    target_layer = (
        model
        .decoder1
        .conv_block
        .norm2
    )


    print(
        "Target layer:",
        "decoder1.conv_block.norm2",
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    image_np = np.asarray(
        image,
        dtype=np.float32,
    )


    image_tensor = (
        torch.from_numpy(
            image_np
        ).float()
    )


    # Swin-UNETR expects:
    #
    # [B, C, D, H, W]

    if image_tensor.ndim == 3:

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif image_tensor.ndim == 4:

        image_tensor = (
            image_tensor
            .unsqueeze(0)
        )


    print(
        "Model input shape:",
        tuple(
            image_tensor.shape
        ),
    )


    # -----------------------------------------------------
    # PROBABILITY VOLUME
    # -----------------------------------------------------

    with torch.no_grad():

        x = image_tensor.to(
            DEVICE
        )


        if x.ndim == 4:

            x = x.unsqueeze(0)


        logits = model(x)


        if isinstance(
            logits,
            (tuple, list),
        ):

            logits = logits[0]


        probabilities = (
            torch.sigmoid(
                logits
            )[0]
            .cpu()
            .numpy()
        )


    # -----------------------------------------------------
    # DISEASE-SPECIFIC POINTS
    # -----------------------------------------------------

    summary = []


    for class_id, disease_name in (
        DISEASES.items()
    ):

        disease_points = [

            p

            for p in points

            if int(
                p["class_id"]
            ) == class_id

        ]


        # -------------------------------------------------
        # NO POINT FOR THIS DISEASE
        # -------------------------------------------------

        if not disease_points:

            print(
                f"No point found for "
                f"{disease_name}"
            )

            continue


        # -------------------------------------------------
        # FIRST MATCHING POINT
        # -------------------------------------------------

        point = (
            disease_points[0]
        )


        print()
        print(
            "-" * 60
        )

        print(
            "Disease:",
            disease_name,
        )


        # -------------------------------------------------
        # PRINT POINT BEFORE XAI
        #
        # This lets us inspect the point produced by
        # the existing Part 2.20B pipeline.
        # -------------------------------------------------

        print(
            "Annotation/model point:",
            (
                point["z"],
                point["y"],
                point["x"],
            ),
        )


        print(
            "Class ID:",
            point["class_id"],
        )


        # -------------------------------------------------
        # POINT-TARGETED XAI
        # -------------------------------------------------

        result = (
            compute_point_targeted_xai(
                model,
                image_tensor,
                point,
                target_layer,
            )
        )


        xai = result[
            "xai"
        ]


        probability = probabilities[
            class_id
        ]


        # -------------------------------------------------
        # LOCALIZATION METRICS
        # -------------------------------------------------

        metrics = (
            evaluate_localization(
                xai,
                point,
            )
        )


        # -------------------------------------------------
        # SUMMARY RECORD
        # -------------------------------------------------

        record = {

            "study_id": STUDY_ID,

            "series_id": SERIES_ID,

            "class_id": class_id,

            "disease": disease_name,

            "target_logit": result[
                "target_logit"
            ],

            "point_z": result[
                "point_z"
            ],

            "point_y": result[
                "point_y"
            ],

            "point_x": result[
                "point_x"
            ],

            "activation_shape": list(
                result[
                    "activation_shape"
                ]
            ),

            "gradient_shape": list(
                result[
                    "gradient_shape"
                ]
            ),

            **metrics,
        }


        summary.append(
            record
        )


        # -------------------------------------------------
        # TERMINAL RESULTS
        # -------------------------------------------------

        print(
            "Target logit:",
            result[
                "target_logit"
            ],
        )


        print(
            "Point:",
            (
                result["point_z"],
                result["point_y"],
                result["point_x"],
            ),
        )


        print(
            "XAI shape:",
            xai.shape,
        )


        print(
            "Activation shape:",
            result[
                "activation_shape"
            ],
        )


        print(
            "Gradient shape:",
            result[
                "gradient_shape"
            ],
        )


        print(
            "Centroid distance:",
            metrics[
                "centroid_distance_voxels"
            ],
        )


        print(
            "R2 concentration:",
            metrics[
                "attribution_concentration_r2"
            ],
        )


        print(
            "R4 concentration:",
            metrics[
                "attribution_concentration_r4"
            ],
        )


        print(
            "R6 concentration:",
            metrics[
                "attribution_concentration_r6"
            ],
        )


        # -------------------------------------------------
        # SAVE RAW XAI
        # -------------------------------------------------

        np.save(
            OUTPUT_DIR
            / f"{disease_name}_xai.npy",
            xai,
        )


        # -------------------------------------------------
        # SAVE VISUALIZATIONS
        # -------------------------------------------------

        save_visualization(
            image_np,
            probability,
            xai,
            point,
            disease_name,
            OUTPUT_DIR,
        )


    # -----------------------------------------------------
    # SAVE SUMMARY
    # -----------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "part46_xai_summary.json"
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


    # -----------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "PART 4.6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    print(
        "No training performed."
    )

    print(
        "No checkpoint modified."
    )

    print(
        "No voxel ground truth fabricated."
    )


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()