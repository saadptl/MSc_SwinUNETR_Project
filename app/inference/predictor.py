from typing import Dict

import torch
import torch.nn.functional as F

from .model_loader import CLASS_NAMES, load_model


def predict_tensor(tensor: torch.Tensor) -> Dict:
    """
    Dashboard inference only.

    Expected tensor:
        [1, 3, 224, 224]

    The tensor must already use the project's training-compatible
    preprocessing.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    if tensor.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor [B,C,H,W], got {tuple(tensor.shape)}"
        )

    if tensor.shape[1] != 3:
        raise ValueError(
            "The trained model expects exactly 3 input channels."
        )

    model = load_model()

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    class_id = int(prediction[0].item())

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "confidence": float(confidence[0].item() * 100.0),
        "probabilities": {
            CLASS_NAMES[i]: float(
                probabilities[0, i].item() * 100.0
            )
            for i in range(len(CLASS_NAMES))
        },
        "logits": [
            float(x)
            for x in logits[0].detach().cpu().tolist()
        ],
        "device": str(device),
    }
