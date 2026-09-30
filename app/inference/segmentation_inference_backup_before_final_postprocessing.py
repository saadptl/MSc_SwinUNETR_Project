
"""
Production-oriented 3D Swin-UNETR segmentation inference.

Important:
- The protected Part104 checkpoint is unchanged.
- The canonical MONAI SwinUNETR architecture is unchanged.
- Inference now uses sliding-window patches matching the training crop
  (32 x 64 x 64) instead of feeding the entire 64 x 96 x 96 volume as one
  patch. This better matches the established training distribution.
- DICOM orientation is interpreted from ImageOrientationPatient and the
  volume is reoriented to a consistent anatomical XYZ convention before
  visualization and model preprocessing. This fixes incorrect coronal and
  sagittal views for non-axial source series.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "segmentation"
    / "rsna_part104_final_checkpoint_selection" / "checkpoints"
    / "final_segmentation_model.pth"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "segmentation"

NUM_CLASSES = 6
MODEL_SHAPE = (64, 96, 96)
TRAINING_ROI = (32, 64, 64)
FEATURE_SIZE = 12
EXPECTED_CHECKPOINT_SHA256 = (
    "fa2ab2eaace098bc7c488332ea1dd7eeb2dee85e3bcfe26956c7785c598fc431"
)

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# Presentation colors. Foreground remains red when "All foreground" is active.
CLASS_COLORS = {
    1: (235, 45, 55),
    2: (60, 145, 245),
    3: (40, 205, 120),
    4: (250, 195, 50),
    5: (175, 95, 245),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _orientation_info(ds: Any) -> Dict[str, Any]:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) < 6:
        return {
            "valid": False,
            "row": np.array([1., 0., 0.]),
            "col": np.array([0., 1., 0.]),
            "normal": np.array([0., 0., 1.]),
        }

    row = np.asarray([float(x) for x in iop[:3]], dtype=np.float64)
    col = np.asarray([float(x) for x in iop[3:6]], dtype=np.float64)
    row /= max(np.linalg.norm(row), 1e-12)
    col /= max(np.linalg.norm(col), 1e-12)
    normal = np.cross(row, col)
    normal /= max(np.linalg.norm(normal), 1e-12)
    return {"valid": True, "row": row, "col": col, "normal": normal}


def _dominant_axis(v: np.ndarray) -> int:
    return int(np.argmax(np.abs(v)))


def _canonicalize_volume(
    volume: np.ndarray,
    datasets: List[Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert [slice,row,column] into canonical [Z,Y,X] where:
      X = patient left/right axis
      Y = patient anterior/posterior axis
      Z = patient inferior/superior axis

    DICOM Patient Position uses LPS coordinates. We preserve the physical
    coordinate ordering and flip axes so the canonical arrays increase in
    +X, +Y, +Z. The exact radiological left/right presentation can be mirrored
    at display time without changing the segmentation coordinates.
    """
    info = _orientation_info(datasets[0])
    if not info["valid"]:
        return volume, {
            "orientation_valid": False,
            "orientation_label": "Unknown",
            "canonicalized": False,
            "axis_map": [0, 1, 2],
            "flips": [False, False, False],
        }

    # Pixel array axes:
    # axis 0 = direction of IOP second triplet (columns of image)
    # axis 1 = direction of IOP first triplet (rows of image)
    # stacked axis = slice normal.
    source_vectors = [
        info["normal"],
        info["col"],
        info["row"],
    ]
    source_axes = [_dominant_axis(v) for v in source_vectors]

    # Need one source axis for each canonical X,Y,Z.
    if len(set(source_axes)) != 3:
        return volume, {
            "orientation_valid": True,
            "orientation_label": "Oblique",
            "canonicalized": False,
            "axis_map": [0, 1, 2],
            "flips": [False, False, False],
        }

    # source_axes[s] tells which patient axis source volume axis s follows.
    # Canonical array order is Z,Y,X => patient axes 2,1,0.
    source_for_target = [
        source_axes.index(2),
        source_axes.index(1),
        source_axes.index(0),
    ]
    out = np.transpose(volume, axes=source_for_target)

    # Determine the physical sign of each target axis.
    target_vectors = [
        source_vectors[source_for_target[0]],
        source_vectors[source_for_target[1]],
        source_vectors[source_for_target[2]],
    ]
    # Target axes are Z,Y,X, so positive patient directions are +Z,+Y,+X.
    target_patient_axes = [2, 1, 0]
    flips = []
    for vec, axis in zip(target_vectors, target_patient_axes):
        flips.append(float(vec[axis]) < 0.0)

    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis=axis)

    normal = info["normal"]
    orientation_label = (
        "Axial" if abs(normal[2]) >= max(abs(normal[0]), abs(normal[1]))
        else "Sagittal" if abs(normal[0]) >= max(abs(normal[1]), abs(normal[2]))
        else "Coronal"
    )

    return np.ascontiguousarray(out), {
        "orientation_valid": True,
        "orientation_label": orientation_label,
        "canonicalized": True,
        "axis_map": source_for_target,
        "flips": flips,
        "row_direction": info["row"].tolist(),
        "column_direction": info["col"].tolist(),
        "slice_normal": normal.tolist(),
    }


def _slice_spacing(datasets: List[Any], normal: np.ndarray) -> float:
    positions = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            p = np.asarray([float(x) for x in ipp[:3]], dtype=np.float64)
            positions.append(float(np.dot(p, normal)))

    if len(positions) >= 2:
        values = np.asarray(positions)
        diffs = np.abs(np.diff(values))
        diffs = diffs[diffs > 1e-4]
        if diffs.size:
            return float(np.median(diffs))

    for ds in datasets:
        for attr in ("SpacingBetweenSlices", "SliceThickness"):
            try:
                v = float(getattr(ds, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    return 1.0


def _dicom_sort_key(ds: Any) -> Tuple:
    info = _orientation_info(ds)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if info["valid"] and ipp is not None and len(ipp) >= 3:
        p = np.asarray([float(x) for x in ipp[:3]], dtype=np.float64)
        return (float(np.dot(p, info["normal"])), int(getattr(ds, "InstanceNumber", 0) or 0))
    return (int(getattr(ds, "InstanceNumber", 0) or 0), 0)


def read_dicom_series(
    series_dir: str | Path,
) -> Tuple[np.ndarray, List[Any], Dict[str, Any]]:
    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    datasets: List[Any] = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        except Exception:
            continue
    if not datasets:
        raise RuntimeError("No readable pixel-bearing DICOM files.")

    datasets.sort(key=_dicom_sort_key)

    arrays = []
    for ds in datasets:
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1.0))
        a += float(getattr(ds, "RescaleIntercept", 0.0))
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = a.max() - a
        arrays.append(a)

    shapes = [a.shape for a in arrays]
    max_h = max(s[0] for s in shapes)
    max_w = max(s[1] for s in shapes)
    volume = np.zeros((len(arrays), max_h, max_w), dtype=np.float32)
    for i, a in enumerate(arrays):
        h, w = a.shape
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        volume[i, y0:y0+h, x0:x0+w] = a

    ori = _orientation_info(datasets[0])
    spacing_xy = getattr(datasets[0], "PixelSpacing", [1.0, 1.0])
    try:
        row_spacing = float(spacing_xy[0])
        col_spacing = float(spacing_xy[1])
    except Exception:
        row_spacing = col_spacing = 1.0

    slice_spacing = _slice_spacing(datasets, ori["normal"])
    canonical, canonical_info = _canonicalize_volume(volume, datasets)

    # Map original [slice,row,col] spacings to canonical [Z,Y,X].
    source_spacing = [slice_spacing, row_spacing, col_spacing]
    if canonical_info["canonicalized"]:
        spacing = [
            source_spacing[canonical_info["axis_map"][0]],
            source_spacing[canonical_info["axis_map"][1]],
            source_spacing[canonical_info["axis_map"][2]],
        ]
    else:
        spacing = source_spacing

    metadata = {
        "series_directory": str(series_dir),
        "num_files_found": len(files),
        "num_readable_slices": len(datasets),
        "original_shapes": [list(s) for s in sorted(set(shapes))],
        "original_shape": list(volume.shape),
        "native_shape": list(canonical.shape),
        "harmonized": len(set(shapes)) > 1,
        "pixel_spacing_mm": [row_spacing, col_spacing],
        "slice_spacing_mm": slice_spacing,
        "voxel_spacing_mm": [float(x) for x in spacing],
        "physical_size_mm": [
            float(canonical.shape[i] * spacing[i]) for i in range(3)
        ],
        "source_orientation": canonical_info["orientation_label"],
        "canonicalized": canonical_info["canonicalized"],
        "orientation": canonical_info,
        "patient_id": str(getattr(datasets[0], "PatientID", "")),
        "study_instance_uid": str(getattr(datasets[0], "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(datasets[0], "SeriesInstanceUID", "")),
        "modality": str(getattr(datasets[0], "Modality", "MR")),
        "rows": int(max_h),
        "columns": int(max_w),
    }
    return canonical, datasets, metadata


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    lo, hi = np.percentile(x[np.isfinite(x)], [1, 99])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((np.clip(x, lo, hi) - lo) / (hi - lo), 0, 1).astype(np.float32)


def resize_3d(volume: np.ndarray, target=MODEL_SHAPE) -> np.ndarray:
    if tuple(volume.shape) == tuple(target):
        return volume.copy()
    t = torch.from_numpy(volume).float()[None, None]
    return F.interpolate(
        t, size=target, mode="trilinear", align_corners=False
    )[0, 0].numpy().astype(np.float32)


def prepare_tensor(volume: np.ndarray) -> torch.Tensor:
    x = normalize_volume(resize_3d(volume))
    return torch.from_numpy(x)[None, None].float().contiguous()


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net", "network", "weights"):
            if isinstance(obj.get(key), dict):
                return {
                    k: v for k, v in obj[key].items()
                    if isinstance(k, str) and torch.is_tensor(v)
                }
        direct = {k: v for k, v in obj.items() if isinstance(k, str) and torch.is_tensor(v)}
        if direct:
            return direct
    raise RuntimeError("No tensor state dictionary found in checkpoint.")


def create_segmentation_model(device: torch.device):
    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
        spatial_dims=3,
    ).to(device)
    count = sum(p.numel() for p in model.parameters())
    if count != 4_078_116:
        raise RuntimeError(f"Canonical parameter count failed: {count}")
    return model


def load_segmentation_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    sha = sha256_file(CHECKPOINT)
    if sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Part104 checkpoint SHA mismatch.\nExpected {EXPECTED_CHECKPOINT_SHA256}\nActual {sha}"
        )

    model = create_segmentation_model(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint), strict=True)
    model.eval()
    return model, device, sha


def predict_volume(model, tensor: torch.Tensor, device: torch.device):
    """
    Inference distribution fix:
    Part98 trained on 32x64x64 foreground-centered crops, so full-volume
    inference now uses overlapping 32x64x64 sliding windows with Gaussian
    blending rather than a single full-volume forward pass.
    """
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=tensor.to(device),
            roi_size=TRAINING_ROI,
            sw_batch_size=1,
            predictor=model,
            overlap=0.50,
            mode="gaussian",
            sigma_scale=0.125,
            sw_device=device,
            device=device,
        )
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)[0]
    return logits, probs[0], pred


def normalize_for_display(image):
    x = np.nan_to_num(np.asarray(image, dtype=np.float32))
    finite = x[np.isfinite(x)]
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    x = np.clip((np.clip(x, lo, hi) - lo) / (hi - lo), 0, 1)
    img = np.uint8(np.power(x, 0.90) * 255)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def make_red_overlay(image, mask, alpha=0.62, boundary=True):
    base = cv2.cvtColor(normalize_for_display(image), cv2.COLOR_GRAY2RGB)
    fg = np.asarray(mask) > 0
    out = base.copy()
    if np.any(fg):
        red = np.zeros_like(base)
        red[..., 0] = 255
        out[fg] = ((1-alpha)*base[fg] + alpha*red[fg]).astype(np.uint8)
        if boundary:
            edge = cv2.morphologyEx(
                fg.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8)
            ) > 0
            out[edge] = (255, 255, 255)
    return out


def make_multiclass_overlay(image, mask, alpha=0.50):
    base = cv2.cvtColor(normalize_for_display(image), cv2.COLOR_GRAY2RGB)
    out = base.copy()
    colors = {
        1:(235,45,55), 2:(60,145,245), 3:(40,205,120),
        4:(250,195,50), 5:(175,95,245)
    }
    for cid, rgb in colors.items():
        region = np.asarray(mask) == cid
        if np.any(region):
            c = np.asarray(rgb, dtype=np.float32)
            out[region] = ((1-alpha)*base[region] + alpha*c).astype(np.uint8)
    return out


def resize_mask_to_native(prediction, native_shape):
    if tuple(prediction.shape) == tuple(native_shape):
        return prediction.astype(np.uint8)
    t = torch.from_numpy(prediction.astype(np.float32))[None, None]
    return F.interpolate(t, size=native_shape, mode="nearest")[0,0].numpy().astype(np.uint8)


def resample_to_isotropic(volume: np.ndarray, spacing_zyx, target_mm=1.0, is_mask=False):
    """Resample a canonical [Z,Y,X] volume into physical isotropic space.

    This is for viewer/MPR geometry. It does not claim to create new acquired
    information; the original DICOM spacing is retained in metadata.
    """
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if np.any(spacing <= 0):
        return volume.copy(), [float(x) for x in spacing]
    target = float(target_mm)
    new_shape = tuple(max(1, int(round(n * sp / target))) for n, sp in zip(volume.shape, spacing))
    if new_shape == tuple(volume.shape):
        return volume.copy(), [target, target, target]
    t = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None]
    mode = "nearest" if is_mask else "trilinear"
    if mode == "nearest":
        out = F.interpolate(t, size=new_shape, mode=mode)[0, 0].numpy()
    else:
        out = F.interpolate(t, size=new_shape, mode=mode, align_corners=False)[0, 0].numpy()
    if is_mask:
        out = np.rint(out).astype(np.uint8)
    else:
        out = out.astype(np.float32)
    return out, [target, target, target]


def representative_indices(depth, count=9):
    if depth <= 1:
        return [0]
    return sorted(set(int(round((depth-1)*f)) for f in np.linspace(0.06, 0.94, count)))


def _save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_hd_views(native_volume, native_mask, spacing_zyx, out_dir, output_size=1400, mpr_target_mm=1.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    mpr_volume, mpr_spacing = resample_to_isotropic(native_volume, spacing_zyx, mpr_target_mm, False)
    mpr_mask, _ = resample_to_isotropic(native_mask, spacing_zyx, mpr_target_mm, True)
    result = {"axial": [], "coronal": [], "sagittal": []}

    planes = {
        "axial": (mpr_volume.shape[0], lambda i: mpr_volume[i], lambda i: mpr_mask[i]),
        "coronal": (mpr_volume.shape[1], lambda i: mpr_volume[:, i, :], lambda i: mpr_mask[:, i, :]),
        "sagittal": (mpr_volume.shape[2], lambda i: mpr_volume[:, :, i], lambda i: mpr_mask[:, :, i]),
    }
    for plane, (depth, get_img, get_mask) in planes.items():
        for i in representative_indices(depth, 5):
            img = get_img(i)
            m = get_mask(i)
            # Preserve aspect ratio; do not force a square medical image.
            disp = normalize_for_display(img)
            overlay = make_red_overlay(img, m)
            h, w = disp.shape
            scale = min(output_size / max(w, 1), output_size / max(h, 1))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            original = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_CUBIC)
            overlay = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_CUBIC)
            p = out_dir / f"{plane}_{i:03d}_overlay_mpr.png"
            _save_rgb(p, overlay)
            po = out_dir / f"{plane}_{i:03d}_mri_mpr.png"
            cv2.imwrite(str(po), original)
            result[plane].append(str(p))
    return result, mpr_volume, mpr_mask, mpr_spacing

def segment_series(series_directory: str | Path, output_name="dashboard_current_segmentation"):
    native_volume, datasets, metadata = read_dicom_series(series_directory)
    input_tensor = prepare_tensor(native_volume)
    model, device, sha = load_segmentation_model()
    logits, probabilities, prediction = predict_volume(model, input_tensor, device)

    pred = prediction.detach().cpu().numpy().astype(np.uint8)
    probs = probabilities.detach().cpu().numpy().astype(np.float32)

    output_dir = OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "segmentation_mask.npy"
    probability_path = output_dir / "class_probabilities.npy"
    np.save(prediction_path, pred)
    np.save(probability_path, probs)

    native_mask = resize_mask_to_native(pred, native_volume.shape)
    views, mpr_volume, mpr_mask, mpr_spacing = save_hd_views(native_volume, native_mask, metadata["voxel_spacing_mm"], output_dir / "visualizations")

    counts = {CLASS_NAMES[i]: int(np.count_nonzero(pred == i)) for i in range(NUM_CLASSES)}
    total = int(pred.size)
    foreground = int(np.count_nonzero(pred > 0))

    return {
        "study_directory": str(series_directory),
        "device": str(device),
        "model_name": "MONAI SwinUNETR (3D)",
        "model_parameters": 4_078_116,
        "model_checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha,
        "input_shape": list(input_tensor.shape),
        "output_shape": list(logits.shape),
        "segmentation_shape": list(pred.shape),
        "native_shape": list(native_volume.shape),
        "native_volume": native_volume,
        "native_mask": native_mask,
        "mpr_volume": mpr_volume,
        "mpr_mask": mpr_mask,
        "mpr_spacing_mm": mpr_spacing,
        "prediction_volume": pred,
        "input_volume": input_tensor[0,0].cpu().numpy(),
        "voxel_counts": counts,
        "foreground_voxels": foreground,
        "foreground_fraction": foreground / total if total else 0.0,
        "mean_class_probabilities": {
            CLASS_NAMES[i]: float(probs[i].mean()) for i in range(NUM_CLASSES)
        },
        "dicom": metadata,
        "prediction_path": str(prediction_path),
        "probability_path": str(probability_path),
        "visualizations": views,
        "slice_indices": representative_indices(native_volume.shape[0], 9),
        "inference_method": "Sliding-window 32×64×64, 50% overlap, Gaussian blending",
        "visualization_mode": "DICOM-orientation-aware physical-space isotropic MPR (1.0 mm) with native labelmap retained",
    }


if __name__ == "__main__":
    print("=" * 88)
    print("SWIN-UNETR PRODUCTION INFERENCE SMOKE TEST")
    print("=" * 88)
    print("Checkpoint exists:", CHECKPOINT.exists())
    print("CUDA available:", torch.cuda.is_available())
    sha = sha256_file(CHECKPOINT)
    print("Checkpoint SHA:", sha)
    print("SHA check:", "PASS" if sha == EXPECTED_CHECKPOINT_SHA256 else "FAIL")
    model, device, _ = load_segmentation_model()
    dummy = torch.zeros((1,1,*MODEL_SHAPE), device=device)
    with torch.inference_mode():
        out = model(dummy)
    print("Model parameters: 4,078,116")
    print("Dummy input:", tuple(dummy.shape))
    print("Model output:", tuple(out.shape))
    assert tuple(out.shape) == (1, NUM_CLASSES, *MODEL_SHAPE)
    print("PASS — PROTECTED SWIN-UNETR CHECKPOINT/ARCHITECTURE")
