"""
MRI Image Preprocessing Pipeline
================================
Utilities to load, preprocess, normalize, and enhance DICOM/PNG/JPG MRI scans
for Swin Transformer and SwinUNETR classification networks.
"""

import io
import numpy as np
from PIL import Image
import torch
import cv2

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def load_mri_image(file_source, filename: str = "") -> Image.Image:
    """Load image from file path or bytes supporting PNG, JPG, and DICOM formats."""
    is_dicom = filename.lower().endswith(".dcm") or (isinstance(file_source, str) and file_source.lower().endswith(".dcm"))
    
    if is_dicom and HAS_PYDICOM:
        try:
            if isinstance(file_source, (bytes, io.BytesIO)):
                if isinstance(file_source, bytes):
                    file_source = io.BytesIO(file_source)
                ds = pydicom.dcmread(file_source)
            else:
                ds = pydicom.dcmread(file_source)
            
            arr = ds.pixel_array.astype(np.float32)
            # Normalize DICOM pixel array to 0-255
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0).astype(np.uint8)
            img = Image.fromarray(arr).convert("RGB")
            return img
        except Exception:
            pass
            
    # Fallback to PIL Image load
    if isinstance(file_source, bytes):
        img = Image.open(io.BytesIO(file_source))
    elif hasattr(file_source, "read"):
        img = Image.open(file_source)
    elif isinstance(file_source, str):
        img = Image.open(file_source)
    else:
        raise ValueError("Unsupported file format or invalid input type.")
        
    return img.convert("RGB")


def preprocess_mri(
    image: Image.Image,
    target_size=(224, 224),
    apply_clahe: bool = True
) -> dict:
    """
    Preprocess MRI image:
    1. Resize to target (224x224)
    2. Convert to Grayscale & apply CLAHE contrast enhancement
    3. Construct RGB normalized tensor and display arrays
    """
    # 1. Resize PIL image
    img_resized = image.resize(target_size, Image.Resampling.BILINEAR)
    np_img = np.array(img_resized)
    
    # 2. Grayscale & Contrast enhancement
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray
        
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # 3. Tensor normalization [0, 1]
    tensor_input = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float() / 255.0
    
    # Standard ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor_normalized = (tensor_input - mean) / std
    tensor_batch = tensor_normalized.unsqueeze(0)  # (1, 3, 224, 224)
    
    return {
        "original_np": np_img,
        "enhanced_np": enhanced_rgb,
        "gray_np": enhanced,
        "tensor_batch": tensor_batch,
        "height": target_size[1],
        "width": target_size[0],
    }
