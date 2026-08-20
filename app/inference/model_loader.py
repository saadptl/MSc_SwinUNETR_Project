import sys
from pathlib import Path

import torch

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

CLASS_NAMES = [
    "Normal/Mild",
    "Moderate",
    "Severe",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_model = None
_checkpoint_info = None


def get_device():
    return DEVICE


def model_exists():
    return MODEL_PATH.exists()


def load_model():
    """
    Load the existing trained checkpoint.

    READ-ONLY:
    - no optimizer
    - no backward pass
    - no training
    - no checkpoint modification
    """
    global _model, _checkpoint_info

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        _checkpoint_info = {
            "type": "training_checkpoint",
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "validation_loss": checkpoint.get("validation_loss"),
            "train_accuracy": checkpoint.get("train_accuracy"),
            "validation_accuracy": checkpoint.get("validation_accuracy"),
            "keys": list(checkpoint.keys()),
        }
    else:
        state_dict = checkpoint
        _checkpoint_info = {
            "type": "state_dict",
            "epoch": None,
            "keys": [],
        }

    model = create_model(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=3,
        pretrained=False,
        dropout=0.30,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    _model = model

    return _model


def get_checkpoint_info():
    if _checkpoint_info is None:
        load_model()
    return _checkpoint_info
