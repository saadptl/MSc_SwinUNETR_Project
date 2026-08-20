from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pydicom
import torch


IMAGE_SIZE = 224


@dataclass
class DICOMSlice:
    path: str
    filename: str
    instance_number: int
    image_position: Optional[float]
    rows: int
    columns: int


def _safe_instance(ds, fallback: int) -> int:
    value = getattr(ds, "InstanceNumber", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_position(ds):
    """
    Prefer ImagePositionPatient[2] for physical slice ordering.
    Return None when unavailable.
    """
    value = getattr(ds, "ImagePositionPatient", None)

    try:
        if value is not None and len(value) >= 3:
            return float(value[2])
    except (TypeError, ValueError):
        pass

    return None


def inspect_dicom(path: Path) -> Dict:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)

    return {
        "filename": path.name,
        "path": str(path),
        "study_instance_uid": str(
            getattr(ds, "StudyInstanceUID", "")
        ),
        "series_instance_uid": str(
            getattr(ds, "SeriesInstanceUID", "")
        ),
        "study_id": str(
            getattr(ds, "StudyID", "")
        ),
        "series_number": str(
            getattr(ds, "SeriesNumber", "")
        ),
        "instance_number": getattr(
            ds, "InstanceNumber", None
        ),
        "modality": str(
            getattr(ds, "Modality", "")
        ),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "patient_position": str(
            getattr(ds, "PatientPosition", "")
        ),
    }


def read_series(directory: str | Path) -> List[DICOMSlice]:
    """
    Read all .dcm files in one directory and sort them.

    Sorting priority:
      1. ImagePositionPatient z-coordinate when available
      2. InstanceNumber
      3. filename
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"DICOM series directory not found:\n{directory}"
        )

    paths = sorted(
        directory.glob("*.dcm"),
        key=lambda p: p.name,
    )

    if not paths:
        raise FileNotFoundError(
            f"No .dcm files found in:\n{directory}"
        )

    slices: List[DICOMSlice] = []

    for fallback, path in enumerate(paths, start=1):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
            )

            slices.append(
                DICOMSlice(
                    path=str(path),
                    filename=path.name,
                    instance_number=_safe_instance(
                        ds,
                        fallback,
                    ),
                    image_position=_safe_position(ds),
                    rows=int(
                        getattr(ds, "Rows", 0)
                    ),
                    columns=int(
                        getattr(ds, "Columns", 0)
                    ),
                )
            )
        except Exception:
            # Invalid DICOM files are skipped rather than crashing
            # the entire dashboard.
            continue

    if not slices:
        raise ValueError(
            "No readable DICOM files were found in the series."
        )

    if all(
        item.image_position is not None
        for item in slices
    ):
        slices.sort(
            key=lambda item: (
                item.image_position,
                item.instance_number,
                item.filename,
            )
        )
    else:
        slices.sort(
            key=lambda item: (
                item.instance_number,
                item.filename,
            )
        )

    return slices


def select_three_adjacent(
    slices: List[DICOMSlice],
    middle_index: Optional[int] = None,
) -> List[DICOMSlice]:
    """
    Select previous / middle / next.

    Edge cases:
      - 1 slice: repeat it three times
      - 2 slices: use first, second, second
      - >=3 slices: select true adjacent slices
    """
    if not slices:
        raise ValueError("No DICOM slices available.")

    if middle_index is None:
        middle_index = len(slices) // 2

    middle_index = max(
        0,
        min(middle_index, len(slices) - 1),
    )

    if len(slices) == 1:
        return [slices[0], slices[0], slices[0]]

    if len(slices) == 2:
        if middle_index == 0:
            return [
                slices[0],
                slices[0],
                slices[1],
            ]

        return [
            slices[0],
            slices[1],
            slices[1],
        ]

    previous_index = max(
        middle_index - 1,
        0,
    )

    next_index = min(
        middle_index + 1,
        len(slices) - 1,
    )

    return [
        slices[previous_index],
        slices[middle_index],
        slices[next_index],
    ]


def _read_pixels(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)

    # Handle MONOCHROME1 convention.
    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":
        image = image.max() - image

    return image


def preprocess_slice(
    image: np.ndarray,
    size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Match the project's established image preprocessing:
      float conversion
      min-max normalization
      uint8 conversion
      CLAHE
      resize
    """
    image = image.astype(np.float32)

    image_min = image.min()
    image_max = image.max()

    image = (
        image - image_min
    ) / (
        image_max - image_min + 1e-8
    )

    image = np.uint8(
        image * 255.0
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    image = clahe.apply(image)

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    return image


def build_model_tensor(
    selected_slices: List[DICOMSlice],
    size: int = IMAGE_SIZE,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Convert previous/middle/next DICOM slices into:

        [1, 3, 224, 224]

    This follows the same channel construction used by the
    training-compatible pipeline.
    """
    if len(selected_slices) != 3:
        raise ValueError(
            "Exactly three slices are required."
        )

    processed = []

    for item in selected_slices:
        raw = _read_pixels(item.path)
        processed.append(
            preprocess_slice(
                raw,
                size=size,
            )
        )

    array = np.stack(
        processed,
        axis=0,
    )

    tensor = torch.tensor(
        array,
        dtype=torch.float32,
    ) / 255.0

    tensor = tensor.unsqueeze(0)

    return tensor, processed


def series_summary(
    slices: List[DICOMSlice],
) -> Dict:
    return {
        "number_of_slices": len(slices),
        "first_file": slices[0].filename,
        "last_file": slices[-1].filename,
        "files": [
            asdict(item)
            for item in slices
        ],
    }


def selected_slice_summary(
    selected: List[DICOMSlice],
) -> Dict:
    return {
        "channel_0_previous": asdict(selected[0]),
        "channel_1_middle": asdict(selected[1]),
        "channel_2_next": asdict(selected[2]),
    }
