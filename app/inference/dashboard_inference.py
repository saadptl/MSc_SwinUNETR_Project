from __future__ import annotations

from typing import Dict

import torch

from inference.dicom_loader import (
    build_model_tensor,
    read_series,
    select_three_adjacent,
    selected_slice_summary,
    series_summary,
)
from inference.predictor import predict_tensor


def analyze_series(
    series_directory: str,
    middle_index: int | None = None,
) -> Dict:
    """
    Complete Phase 3 Part 3 inference preparation.

    This function:
      1. discovers DICOM files
      2. sorts the series
      3. selects previous/middle/next
      4. applies training-compatible preprocessing
      5. builds [1,3,224,224]
      6. runs the existing trained classifier

    No training is performed.
    """
    slices = read_series(series_directory)

    selected = select_three_adjacent(
        slices,
        middle_index=middle_index,
    )

    tensor, processed_images = build_model_tensor(
        selected
    )

    prediction = predict_tensor(tensor)

    return {
        "series": series_summary(slices),
        "selected": selected_slice_summary(selected),
        "tensor_shape": list(tensor.shape),
        "prediction": prediction,
        "processed_images": processed_images,
        "tensor": tensor,
    }
