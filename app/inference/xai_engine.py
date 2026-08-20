from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inference.model_loader import CLASS_NAMES, load_model


def _extract_tensor(output):
    """Extract a tensor from common Swin feature-output structures."""
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        tensors = [
            item for item in output
            if isinstance(item, torch.Tensor)
        ]
        if tensors:
            return tensors[-1]

    if isinstance(output, dict):
        for key in (
            "last_hidden_state",
            "features",
            "x",
            "output",
        ):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(
        "Could not extract a feature tensor from the model output."
    )


def _get_feature_forward(model):
    """
    Locate the feature-extraction path used by the existing model.

    The project has previously demonstrated a feature representation
    of [B, 7, 7, 768]. We prefer the model's own forward_features path.
    """
    candidates = [
        (model, "forward_features"),
        (getattr(model, "backbone", None), "forward_features"),
        (getattr(model, "swin", None), "forward_features"),
    ]

    for owner, name in candidates:
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)

    raise AttributeError(
        "No forward_features method was found on the loaded model. "
        "The XAI engine will not guess a target layer."
    )


def _classifier(model):
    classifier = getattr(model, "classifier", None)

    if classifier is None:
        raise AttributeError(
            "The loaded model does not expose model.classifier."
        )

    return classifier


def _pool_features(features: torch.Tensor) -> torch.Tensor:
    """
    Convert [B,H,W,C] or [B,C,H,W] to [B,C].

    For the validated Swin representation [B,7,7,768],
    mean pooling is applied over the two spatial dimensions.
    """
    if features.ndim == 4:
        if features.shape[-1] == 768:
            return features.mean(dim=(1, 2))

        if features.shape[1] == 768:
            return features.mean(dim=(2, 3))

    if features.ndim == 3:
        return features.mean(dim=1)

    if features.ndim == 2:
        return features

    raise ValueError(
        f"Unsupported feature shape: {tuple(features.shape)}"
    )


def _features_to_bhw(features: torch.Tensor) -> torch.Tensor:
    """Return feature map as [B,H,W,C]."""
    if features.ndim != 4:
        raise ValueError(
            "Grad-CAM requires a 4D spatial feature map."
        )

    if features.shape[-1] >= features.shape[1]:
        return features

    return features.permute(0, 2, 3, 1).contiguous()


def _manual_logits(model, features: torch.Tensor) -> torch.Tensor:
    pooled = _pool_features(features)
    classifier = _classifier(model)

    # Preserve the classifier's own forward behavior, including bias.
    return classifier(pooled)


def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().float().cpu().numpy()

    cam = np.maximum(cam, 0.0)

    minimum = float(cam.min())
    maximum = float(cam.max())

    if maximum - minimum < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)

    cam = (cam - minimum) / (
        maximum - minimum
    )

    return cam.astype(np.float32)


def _cam_from_features(
    features: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    """
    Grad-CAM:
        weights = global-average-pool(gradients)
        cam = ReLU(sum(weights * features))
    """
    feature_map = _features_to_bhw(features)
    grad_map = _features_to_bhw(gradients)

    weights = grad_map.mean(
        dim=(1, 2),
        keepdim=True,
    )

    cam = (
        feature_map * weights
    ).sum(dim=-1)

    cam = F.relu(cam)

    return _normalize_cam(cam[0])


def generate_gradcam(
    tensor: torch.Tensor,
    target_class: int | None = None,
    original_prediction: Dict | None = None,
) -> Dict:
    """
    Generate Grad-CAM for the existing trained model.

    No optimizer, no parameter update and no checkpoint write occur.

    A forward-output consistency check is performed before XAI.
    If the manually reconstructed classifier path does not reproduce
    model(x), the function aborts instead of producing a potentially
    misleading heatmap.
    """
    model = load_model()

    model.eval()

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    if tensor.ndim != 4 or tensor.shape[1:] != (3, 224, 224):
        raise ValueError(
            "Grad-CAM expects tensor shape [1,3,224,224]. "
            f"Received {tuple(tensor.shape)}."
        )

    # Keep the original model output for the target class.
    with torch.no_grad():
        original_logits = model(tensor)

    if original_prediction is None:
        probabilities = F.softmax(
            original_logits,
            dim=1,
        )
        target_class = int(
            probabilities.argmax(dim=1)[0].item()
        )
    else:
        target_class = int(
            original_prediction["class_id"]
        )

    feature_forward = _get_feature_forward(model)

    # The input must participate in autograd for Grad-CAM.
    x = tensor.detach().clone().requires_grad_(True)

    features = _extract_tensor(
        feature_forward(x)
    )

    if not features.requires_grad:
        raise RuntimeError(
            "Feature tensor does not require gradients. "
            "Grad-CAM cannot be generated."
        )

    manual_logits = _manual_logits(
        model,
        features,
    )

    # Verify the reconstructed feature->classifier path.
    max_logit_error = float(
        torch.max(
            torch.abs(
                original_logits.detach()
                - manual_logits.detach()
            )
        ).item()
    )

    if max_logit_error > 1e-4:
        raise RuntimeError(
            "XAI safety check failed: the feature/classifier path "
            f"does not reproduce model output. Max logit error = "
            f"{max_logit_error:.8f}. No heatmap was generated."
        )

    score = manual_logits[0, target_class]

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=features,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    cam = _cam_from_features(
        features,
        gradients,
    )

    return {
        "target_class": target_class,
        "target_class_name": CLASS_NAMES[target_class],
        "cam": cam,
        "cam_shape": tuple(cam.shape),
        "cam_min": float(cam.min()),
        "cam_max": float(cam.max()),
        "cam_mean": float(cam.mean()),
        "cam_std": float(cam.std()),
        "feature_shape": tuple(features.shape),
        "max_logit_error": max_logit_error,
        "logits": original_logits[0].detach().cpu().numpy(),
    }


def make_heatmap(
    cam: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    heatmap = cv2.resize(
        cam,
        output_size,
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0.0, 1.0) * 255.0
    )

    return cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )


def make_overlay(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      heatmap RGB
      overlay RGB
    """
    if image.ndim != 2:
        raise ValueError(
            "Expected a single-channel grayscale image."
        )

    if image.dtype != np.uint8:
        image = np.uint8(
            np.clip(image, 0.0, 255.0)
        )

    height, width = image.shape[:2]

    heatmap_bgr = make_heatmap(
        cam,
        (width, height),
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )

    overlay_bgr = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        heatmap_bgr,
        alpha,
        0.0,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return heatmap_rgb, overlay_rgb


def save_xai_outputs(
    xai_result: Dict,
    middle_image: np.ndarray,
    output_directory: str | Path,
    stem: str = "dashboard_xai",
) -> Dict[str, str]:
    output_directory = Path(output_directory)

    heatmap_dir = output_directory / "heatmaps"
    overlay_dir = output_directory / "overlays"
    panel_dir = output_directory / "panels"

    for directory in (
        heatmap_dir,
        overlay_dir,
        panel_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    heatmap_rgb, overlay_rgb = make_overlay(
        middle_image,
        xai_result["cam"],
    )

    heatmap_path = heatmap_dir / f"{stem}_gradcam.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"

    cv2.imwrite(
        str(heatmap_path),
        cv2.cvtColor(
            heatmap_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    return {
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }
