from pathlib import Path
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part33_balanced_symmetry_refinement"
    / "checkpoints"
    / "part33_best_development_macro.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part45_3d_xai"
)

STUDY_ID = "7143189"
SERIES_ID = "3219733239"

MODEL_SHAPE = (64, 96, 96)
NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}


def load_model(device):
    print("Building Swin-UNETR...")

    model = part220b.build_model()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    cleaned = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    result = model.load_state_dict(
        cleaned,
        strict=True,
    )

    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: "
            f"missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )

    model = model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


def find_decoder_target(model):
    """
    Select the highest-resolution decoder feature module.

    We use decoder1's final normalization layer if available.
    This is a real model feature, not a synthetic activation.
    """

    candidates = [
        "decoder1.conv_block.norm2",
        "decoder1.conv_block",
        "decoder1",
    ]

    modules = dict(model.named_modules())

    for name in candidates:
        if name in modules:
            print(f"XAI target layer: {name}")
            return modules[name], name

    raise RuntimeError(
        "Could not locate a suitable decoder1 target layer."
    )


def normalize_attribution(x):
    x = x.detach().float()

    x = F.relu(x)

    minimum = x.amin()
    maximum = x.amax()

    if float(maximum - minimum) < 1e-12:
        return torch.zeros_like(x)

    return (x - minimum) / (maximum - minimum)


def main():
    print("=" * 80)
    print("PART 4.5")
    print("REAL 3D SWIN-UNETR XAI PROTOTYPE")
    print("=" * 80)

    print()
    print("No checkpoint modification.")
    print("No training.")
    print("No synthetic activation.")
    print("No fabricated voxel masks.")
    print()

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------
    # Load real physical-space case
    # ------------------------------------------------------------------

    manifest = part220b.load_manifest()

    point_df = manifest[
        (manifest["study_id"].astype(str) == STUDY_ID)
        & (manifest["series_id"].astype(str) == SERIES_ID)
    ].copy()

    if point_df.empty:
        raise RuntimeError(
            "Study/series was not found in the point-supervision manifest."
        )

    loaded = part220b.load_case(
        STUDY_ID,
        SERIES_ID,
        point_df,
    )

    if len(loaded) == 3:
        image, points, geometry = loaded
    elif len(loaded) == 4:
        image, points, geometry, _ = loaded
    else:
        raise RuntimeError(
            f"Unexpected load_case() return length: {len(loaded)}"
        )

    print()
    print("CASE")
    print(f"Study ID : {STUDY_ID}")
    print(f"Series ID: {SERIES_ID}")
    print(f"Image shape: {tuple(image.shape)}")

    # ------------------------------------------------------------------
    # Tensor
    # ------------------------------------------------------------------

    image_np = np.asarray(
        image,
        dtype=np.float32,
    )

    if image_np.shape != MODEL_SHAPE:
        raise RuntimeError(
            f"Unexpected model volume shape: "
            f"{image_np.shape}; expected {MODEL_SHAPE}"
        )

    tensor = torch.from_numpy(
        image_np
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = load_model(device)

    target_layer, target_name = find_decoder_target(model)

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            activations["value"] = output
            output.retain_grad()

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    tensor.requires_grad_(True)

    print()
    print("Running model forward pass...")

    logits = model(tensor)

    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(
            f"Unexpected model output type: {type(logits)}"
        )

    if logits.ndim != 5:
        raise RuntimeError(
            f"Expected [B,C,D,H,W], got {tuple(logits.shape)}"
        )

    if logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} classes, "
            f"got {logits.shape[1]}"
        )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    # ------------------------------------------------------------------
    # Disease selection
    #
    # Run XAI independently for all five disease channels.
    # ------------------------------------------------------------------

    results = {}

    for class_id in range(1, NUM_CLASSES):

        class_name = CLASS_NAMES[class_id]

        print()
        print(
            f"Generating attribution: "
            f"{class_id} — {class_name}"
        )

        model.zero_grad(
            set_to_none=True
        )

        if tensor.grad is not None:
            tensor.grad.zero_()

        target_layer.zero_grad(
            set_to_none=True
        )

        # Aggregate the selected disease probability across
        # the complete volume.
        #
        # This asks:
        # "Which spatial features increase this disease's
        # predicted probability?"
        target_score = probabilities[
            0,
            class_id,
        ].mean()

        target_score.backward(
            retain_graph=True
        )

        if "value" not in activations:
            raise RuntimeError(
                "Target-layer activation was not captured."
            )

        feature = activations["value"]

        if feature.grad is None:
            raise RuntimeError(
                "Target-layer gradient was not captured."
            )

        print(
            f"Feature shape: "
            f"{tuple(feature.shape)}"
        )

        if feature.ndim != 5:
            raise RuntimeError(
                "3D XAI requires a 5D feature tensor "
                "[B,C,D,H,W]."
            )

        # 3D Grad-CAM weights:
        # average gradient across D,H,W.
        weights = feature.grad.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = (
            feature * weights
        ).sum(dim=1, keepdim=True)

        cam = normalize_attribution(
            cam
        )

        # Resize attribution to model input volume.
        cam = F.interpolate(
            cam,
            size=MODEL_SHAPE,
            mode="trilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        probability = probabilities[
            0,
            class_id,
        ].detach().cpu().numpy()

        results[class_name] = {
            "class_id": class_id,
            "class_name": class_name,
            "cam": cam,
            "probability": probability,
            "cam_max": float(cam.max()),
            "cam_mean": float(cam.mean()),
            "cam_std": float(cam.std()),
            "probability_max": float(probability.max()),
            "probability_mean": float(probability.mean()),
        }

    handle.remove()

    # ------------------------------------------------------------------
    # Save raw XAI volumes
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_arrays = {}

    metadata = {
        "study_id": STUDY_ID,
        "series_id": SERIES_ID,
        "checkpoint": str(CHECKPOINT),
        "model_shape": list(MODEL_SHAPE),
        "target_layer": target_name,
        "method": "3D Grad-CAM-style gradient attribution",
        "note": (
            "Model-derived attribution. "
            "Not a voxel-level ground-truth mask."
        ),
    }

    for class_name, result in results.items():

        safe_name = (
            class_name
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        save_arrays[
            f"{safe_name}_xai"
        ] = result["cam"]

        save_arrays[
            f"{safe_name}_probability"
        ] = result["probability"]

        metadata[
            safe_name
        ] = {
            "class_id": result["class_id"],
            "cam_max": result["cam_max"],
            "cam_mean": result["cam_mean"],
            "cam_std": result["cam_std"],
            "probability_max": result["probability_max"],
            "probability_mean": result["probability_mean"],
        }

    np.savez_compressed(
        OUTPUT_DIR / "part45_3d_xai_volumes.npz",
        **save_arrays,
    )

    with open(
        OUTPUT_DIR / "part45_3d_xai_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("PART 4.5 COMPLETE")
    print("=" * 80)
    print(f"Target layer: {target_name}")
    print(
        f"XAI volume: "
        f"{OUTPUT_DIR / 'part45_3d_xai_volumes.npz'}"
    )
    print(
        f"Metadata: "
        f"{OUTPUT_DIR / 'part45_3d_xai_metadata.json'}"
    )
    print()
    print(
        "The attribution maps are model-derived "
        "and are NOT ground-truth segmentation masks."
    )


if __name__ == "__main__":
    main()
