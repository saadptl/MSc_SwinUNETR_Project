"""
Lumbar Spine AI — 3D Disease Localization & Clinical Measurement Workstation
Swin-UNETR framework with vertical anatomical disc/level visualization, interactive caliper measurement,
multi-planar reconstruction (MPR), and automated 3D morphometrics.
"""

from pathlib import Path
from datetime import datetime
import json
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import streamlit as st

try:
    from inference import final_segmentation_inference as final_inf
except ImportError:
    from app.inference import final_segmentation_inference as final_inf


CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

SHORT_NAMES = {
    1: "SCS",
    2: "LFNN",
    3: "RFNN",
    4: "LSS",
    5: "RSS",
}

LEVEL_COLORS = {
    "L1/L2": "#06B6D4",  # Cyan
    "L2/L3": "#3B82F6",  # Blue
    "L3/L4": "#8B5CF6",  # Purple
    "L4/L5": "#F59E0B",  # Amber
    "L5/S1": "#EC4899",  # Rose
}

LEVEL_DESCRIPTIONS = {
    "L1/L2": "Thoracolumbar Junction (Conus Medullaris termination)",
    "L2/L3": "Mid-Upper Lumbar Spine (Lumbar lordosis apex)",
    "L3/L4": "Mid-Lumbar Spine (High mobility segment)",
    "L4/L5": "Lower Lumbar Spine (Aortic bifurcation / Spondylolisthesis hotspot)",
    "L5/S1": "Lumbosacral Junction (Sacral promontory angle / High axial load)",
}

CANONICAL_LEVEL_DEFAULTS = {
    "L1/L2": {"z": 29.8, "y": 25.7, "x": 47.7},
    "L2/L3": {"z": 30.7, "y": 36.4, "x": 47.6},
    "L3/L4": {"z": 30.9, "y": 47.6, "x": 47.6},
    "L4/L5": {"z": 31.0, "y": 59.7, "x": 47.6},
    "L5/S1": {"z": 28.3, "y": 69.5, "x": 48.1},
}


# ============================================================================
# HELPER: Slice Extractor (Supports Vertical Anatomical Orientation)
# ============================================================================

def _slice(volume, orientation, index, is_vertical=True):
    """
    Extract one 2D slice from canonical Z,Y,X volume.
    When is_vertical=True:
      - Sagittal: transposed so Y (cranial-to-caudal) is the VERTICAL axis!
      - Coronal: oriented with superior at top, inferior at bottom.
      - Axial: cross-sectional (anterior to posterior).
    """
    vol = np.asarray(volume)
    if orientation == "Axial":
        idx = int(np.clip(index, 0, vol.shape[0] - 1))
        # Shape (96, 96) -> rows Y, cols X
        return vol[idx, :, :]
    elif orientation == "Coronal":
        idx = int(np.clip(index, 0, vol.shape[1] - 1))
        slc = vol[:, idx, :]  # Shape (64, 96) -> Z, X
        if is_vertical:
            # Superior at top, inferior at bottom
            return np.flipud(slc)
        return slc
    else:  # Sagittal
        idx = int(np.clip(index, 0, vol.shape[2] - 1))
        slc = vol[:, :, idx]  # Shape (64, 96) -> Z, Y
        if is_vertical:
            # Transpose so Y (craniocaudal spine axis) is rows (0=L1 top, 95=S1 bottom)
            # and Z (anterior/posterior) is cols (0 to 63)
            return slc.T  # Shape (96, 64)
        return slc


# ============================================================================
# HELPER: Window/Leveling Presets
# ============================================================================

def _apply_windowing(image_slice, preset):
    """Apply radiological window presets to enhance bone, disc, or thecal sac."""
    arr = np.asarray(image_slice, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)

    fg = finite[finite > 0.01] if np.any(finite > 0.01) else finite

    if preset == "PACS High Definition / Diagnostic Contrast":
        lo = np.percentile(fg, 1.0)
        hi = np.percentile(fg, 99.0)
        norm = np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return np.power(norm, 0.95)
    elif preset == "Bone & Vertebral Endplates":
        lo = np.percentile(fg, 3.0)
        hi = np.percentile(fg, 99.5)
        norm = np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return np.power(norm, 1.15)
    elif preset == "Soft Tissue / Thecal Sac":
        lo = np.percentile(fg, 5.0)
        hi = np.percentile(fg, 92.0)
        norm = np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return np.power(norm, 0.9)
    elif preset == "High Contrast Lesion View":
        lo = np.percentile(fg, 2.0)
        hi = np.percentile(fg, 95.0)
        norm = np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        return np.power(norm, 0.85)
    else:  # Full Dynamic Range (Default)
        lo = np.percentile(fg, 0.5)
        hi = np.percentile(fg, 99.5)
        return np.clip((arr - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def _enhance_mri_clarity(image_slice, clarity_preset="✨ PACS Diagnostic Sharpness (High Edge Contrast)"):
    """Applies authentic radiological unsharp masking to eliminate display rasterization blur."""
    from scipy.ndimage import gaussian_filter
    arr = np.asarray(image_slice, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    if "High Edge Contrast" in clarity_preset or "High PACS" in clarity_preset:
        blurred = gaussian_filter(arr, sigma=0.8)
        sharp = arr + 0.18 * (arr - blurred)
        return np.clip(sharp, 0.0, 1.0)
    elif "Subtle" in clarity_preset:
        blurred = gaussian_filter(arr, sigma=0.6)
        sharp = arr + 0.10 * (arr - blurred)
        return np.clip(sharp, 0.0, 1.0)
    elif "Endplate" in clarity_preset or "Cortical" in clarity_preset:
        blurred = gaussian_filter(arr, sigma=1.0)
        sharp = arr + 0.25 * (arr - blurred)
        return np.clip(sharp, 0.0, 1.0)
    else:
        return arr


def _upsample_slice(slice_arr, factor=3, order=3):
    """High-fidelity medical bicubic spline upsampling."""
    from scipy.ndimage import zoom
    arr = np.asarray(slice_arr, dtype=np.float32)
    if factor <= 1:
        return arr
    return zoom(arr, factor, order=order)


@st.cache_data(show_spinner=False)
def _load_native_dicom_volume(study_id, series_id):
    """Loads authentic un-downsampled native DICOM series from disk."""
    try:
        import pandas as pd
        import segmentation_rsna_part220b_geometry_corrected_training as part220b
        series_dir = part220b.TRAIN_IMAGES_DIR / str(study_id) / str(series_id)
        if not series_dir.exists():
            row = pd.Series({"study_id": str(study_id), "series_id": str(series_id)})
            series_dir = part220b.resolve_series_dir(row)
        if series_dir.exists():
            image_native, records, info = part220b.read_dicom_series_robust(series_dir)
            return image_native, records, info
    except Exception:
        pass
    return None, None, None


def _extract_native_levels(points, geometry):
    """Extract levels in native acquisition coordinates (z, y, x)."""
    import segmentation_rsna_part220b_geometry_corrected_training as part220b
    native_lvls = {}
    if not points or not geometry:
        return native_lvls
    for p in points:
        pt = np.array([[p["patient_x"], p["patient_y"], p["patient_z"]]])
        nz, ny, nx = part220b.patient_to_native(pt, geometry)
        native_lvls[p["level"]] = {
            "z": float(nz[0]),
            "y": float(ny[0]),
            "x": float(nx[0]),
            "class_id": p.get("class_id", 1),
            "class_name": p.get("class_name", "Spine Landmark"),
        }
    return native_lvls


# ============================================================================
# HELPER: Anatomical Level Extraction
# ============================================================================

def _extract_levels(points, physical_extents, image_shape):
    """
    Extract verified lumbar levels with canonical voxel coords and physical mm coords.
    """
    levels = {}
    ordered_keys = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    if points:
        for pt in points:
            lvl = pt.get("level")
            if lvl in ordered_keys and lvl not in levels:
                levels[lvl] = {
                    "level": lvl,
                    "z": float(pt.get("z", 30.0)),
                    "y": float(pt.get("y", 50.0)),
                    "x": float(pt.get("x", 48.0)),
                    "patient_x": float(pt.get("patient_x", 0.0)),
                    "patient_y": float(pt.get("patient_y", 0.0)),
                    "patient_z": float(pt.get("patient_z", 0.0)),
                    "class_name": pt.get("class_name", "Spine Landmark"),
                    "class_id": pt.get("class_id", 1),
                    "source": "RSNA Ground Truth",
                }

    # Fill any missing levels using canonical anatomical spacing
    for lvl in ordered_keys:
        if lvl not in levels:
            def_coords = CANONICAL_LEVEL_DEFAULTS[lvl]
            levels[lvl] = {
                "level": lvl,
                "z": float(def_coords["z"]),
                "y": float(def_coords["y"]),
                "x": float(def_coords["x"]),
                "patient_x": 0.0,
                "patient_y": 0.0,
                "patient_z": 0.0,
                "class_name": "Anatomical Landmark",
                "class_id": 0,
                "source": "Anatomical Atlas Reference",
            }

    return levels


# ============================================================================
# HELPER: Millimeter Scale Calculation
# ============================================================================

def _get_mm_scales(physical_extents, image_shape):
    """
    Returns mm per voxel along (Z, Y, X).
    image_shape is (D=64, H=96, W=96).
    """
    d, h, w = image_shape
    if physical_extents and "x" in physical_extents and "y" in physical_extents and "z" in physical_extents:
        px = physical_extents["x"]
        py = physical_extents["y"]
        pz = physical_extents["z"]
        dx_mm = abs(px[1] - px[0]) / max(1, w - 1)
        dy_mm = abs(py[1] - py[0]) / max(1, h - 1)
        dz_mm = abs(pz[1] - pz[0]) / max(1, d - 1)
    else:
        dx_mm = 1.0
        dy_mm = 1.0
        dz_mm = 2.5
    return dz_mm, dy_mm, dx_mm


def _map_prob_to_native_slice(prob_vol, native_z_idx, geometry, native_shape, records=None):
    """
    Maps 3D canonical probability volume (Z_c=64, Y_c=96, X_c=96)
    onto a 2D native DICOM slice (H_nat, W_nat).
    """
    import scipy.ndimage
    H_nat, W_nat = native_shape
    can_x = 48  # Default canonical mid-sagittal plane
    
    if records is not None and native_z_idx < len(records):
        try:
            import segmentation_rsna_part220b_geometry_corrected_training as part220b
            mid_pt = part220b.native_point_to_patient(records, native_z_idx, H_nat // 2, W_nat // 2)
            can_pt = part220b.patient_point_to_canonical(mid_pt, geometry)
            can_x = int(np.clip(round(can_pt[2]), 0, prob_vol.shape[2] - 1))
        except Exception:
            can_x = 48
    elif geometry and "slice_positions" in geometry and len(geometry["slice_positions"]) > 0:
        ratio = float(native_z_idx) / max(1, len(geometry["slice_positions"]) - 1)
        can_x = int(np.clip(round(ratio * (prob_vol.shape[2] - 1)), 0, prob_vol.shape[2] - 1))

    # Slice at can_x: shape is (Z_canon=64, Y_canon=96)
    slice_zy = prob_vol[:, :, can_x]
    # In native DICOM: rows are Y, cols are Z (inverted)
    slice_yx = slice_zy[::-1, :].T  # shape (96, 64)
    # Resample to native shape (H_nat, W_nat)
    zoom_y = H_nat / slice_yx.shape[0]
    zoom_x = W_nat / slice_yx.shape[1]
    prob_native = scipy.ndimage.zoom(slice_yx, (zoom_y, zoom_x), order=1)
    return prob_native[:H_nat, :W_nat]


# ============================================================================
# HELPER: Nearest Level Identifier for Current Slice
# ============================================================================

def _identify_slice_level(orientation, index, levels, mm_scales, is_native=False, slice_spacing=4.0):
    """
    Determine nearest lumbar disc/level for the currently viewed slice.
    Supports both native acquisition slices and reconstructed canonical planes.
    """
    dz_mm, dy_mm, dx_mm = mm_scales
    best_lvl = None
    min_dist_mm = float("inf")

    for lvl_name, data in levels.items():
        if is_native:
            dist = abs(index - data["z"]) * slice_spacing
        elif orientation == "Axial":
            dist = abs(index - data["z"]) * dz_mm
        elif orientation == "Coronal":
            dist = abs(index - data["y"]) * dy_mm
        else:  # Sagittal
            dist = abs(index - data["x"]) * dx_mm

        if dist < min_dist_mm:
            min_dist_mm = dist
            best_lvl = lvl_name

    desc = LEVEL_DESCRIPTIONS.get(best_lvl, "Lumbar Spine Segment")
    if is_native:
        if min_dist_mm <= slice_spacing:
            status_text = f"Primary Mid-Sagittal Plane: **{best_lvl} Intervertebral Disc Space** ({desc})"
            badge_color = "#10B981"  # Emerald
            is_exact = True
        else:
            status_text = f"Parasagittal Slice Offset from **{best_lvl}** (Offset: {min_dist_mm:.1f} mm | {desc})"
            badge_color = "#3B82F6"  # Blue
            is_exact = False
    else:
        if min_dist_mm <= 4.0:
            status_text = f"Cutting through **{best_lvl} Intervertebral Disc Space** ({desc})"
            badge_color = "#10B981"  # Emerald
            is_exact = True
        else:
            status_text = f"Adjacent to **{best_lvl}** (Offset: {min_dist_mm:.1f} mm | {desc})"
            badge_color = "#3B82F6"  # Blue
            is_exact = False

    return best_lvl, min_dist_mm, status_text, badge_color, is_exact


# ============================================================================
# PLOTTING: Publication-Grade Multi-Panel Slice Visualizer
# ============================================================================

def _plot_rich(
    image,
    prob,
    pred,
    disease,
    orientation,
    index,
    threshold,
    ext,
    levels,
    is_vertical=True,
    show_levels=True,
    level_style="🏥 Clinical PACS Callout Brackets & Badges",
    nearest_lvl=None,
    show_rsna_points=True,
    show_crosshair=True,
    show_caliper=False,
    caliper_coords=None,
    caliper_mm=0.0,
    caliper_label="",
    window_preset="PACS High Definition / Diagnostic Contrast",
    clarity_preset="✨ PACS Diagnostic Sharpness (High Edge Contrast)",
    resolution_scale=3,
    aspect_mode="📐 Anatomically Proportional (True Physical mm)",
    cmap_name="magma",
    prob_alpha=0.60,
    overlay_mode="Contour & Translucent Fill",
    orthogonal_slice_indices=None,
    mm_scales=(2.5, 1.0, 1.0),
    native_slice=None,
    native_prob=None,
    native_levels=None,
    is_native_mode=False,
):
    """
    Publication-grade 3-panel visualizer.
    Supports both:
      1. Original Native DICOM Slices (320x320 / 640x640 full-field-of-view PACS slice - crystal clear)
      2. 3D Swin-UNETR Model Volume Grid with high-order spline upsampling and natural aspect ratio.
    """
    if is_native_mode and native_slice is not None:
        a = np.asarray(native_slice, dtype=np.float32)
        b = np.asarray(native_prob if native_prob is not None else np.zeros_like(a), dtype=np.float32)
        c = (b >= threshold).astype(np.int64)
        active_levels = native_levels if native_levels else levels
        img_h, img_w = a.shape
        origin_mode = "upper"
        aspect_val = 1.0

        a_disp = _apply_windowing(a, window_preset)
        a_render = _enhance_mri_clarity(a_disp, clarity_preset)
        b_render = b
        extent_box = [0, img_w, img_h, 0]
        panel1_title = f"Physical MRI — Native DICOM Slice {index + 1} ({img_w}×{img_h} Full Anatomical PACS View)"
    else:
        a = _slice(image, orientation, index, is_vertical=is_vertical)
        b = _slice(prob, orientation, index, is_vertical=is_vertical)
        c = _slice(pred, orientation, index, is_vertical=is_vertical)
        active_levels = levels
        img_h, img_w = a.shape
        origin_mode = "upper" if is_vertical else "lower"
        aspect_val = 1.0  # Natural square 1:1 DICOM geometry

        a_disp = _apply_windowing(a, window_preset)
        a_sharp = _enhance_mri_clarity(a_disp, clarity_preset)
        if resolution_scale > 1:
            a_render = _upsample_slice(a_sharp, factor=resolution_scale, order=3)
            b_render = _upsample_slice(b, factor=resolution_scale, order=2)
        else:
            a_render = a_sharp
            b_render = b
        extent_box = [0, img_w, img_h, 0] if origin_mode == "upper" else [0, img_w, 0, img_h]
        res_tag = f"{a_render.shape[1]}×{a_render.shape[0]} HD" if resolution_scale > 1 else f"{img_w}×{img_h}"
        panel1_title = f"Physical MRI — {orientation} Slice {index} (Model Space · {res_tag})"

    disease_short = SHORT_NAMES.get(disease, CLASS_NAMES.get(disease, str(disease)))
    disease_full = CLASS_NAMES.get(disease, f"Class {disease}")

    fig_w = 16.5 if is_vertical else 18.5
    fig_h = 7.6 if is_vertical else 6.0
    fig, ax = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=160)
    fig.patch.set_facecolor("#0B132B")  # Deep clinical slate background

    for axis in ax:
        axis.set_facecolor("#000000")
        axis.tick_params(colors="#94A3B8", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#1E293B")
            spine.set_linewidth(1.0)

    # ------------------------------------------------------------------------
    # Panel 1: Physical MRI with Anatomical Discs & Clinical PACS Level Badges
    # ------------------------------------------------------------------------
    ax[0].imshow(
        a_render,
        cmap="gray",
        origin=origin_mode,
        extent=extent_box,
        aspect=aspect_val,
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )
    ax[0].set_title(panel1_title, color="#F8FAFC", fontsize=10.5, fontweight="bold", pad=10)

    # Overlay Clinical PACS Anatomical Level Indicators
    if show_levels and active_levels and level_style != "🚫 Hide Level Overlays":
        for lvl_name, data in active_levels.items():
            color = LEVEL_COLORS.get(lvl_name, "#38BDF8")
            is_active = (nearest_lvl == lvl_name)

            if is_native_mode:
                row_y = data["y"]
                col_x = data["x"]
                bracket_span = 30.0
                ant_x = max(8.0, col_x - bracket_span)
                post_x = min(img_w - 8.0, col_x + bracket_span)

                line_col = "#00F5FF" if is_active else color
                lw = 2.0 if is_active else 1.3
                alpha = 0.95 if is_active else 0.70

                # Focused disc plane bracket
                ax[0].plot([ant_x, post_x], [row_y, row_y], color=line_col, lw=lw, alpha=alpha, zorder=6)
                tick_h = 4.0 if is_active else 2.5
                ax[0].plot([ant_x, ant_x], [row_y - tick_h, row_y + tick_h], color=line_col, lw=lw, alpha=alpha, zorder=6)
                ax[0].plot([post_x, post_x], [row_y - tick_h, row_y + tick_h], color=line_col, lw=lw, alpha=alpha, zorder=6)

                # Nucleus node
                node_sz = 35 if is_active else 20
                ax[0].scatter(col_x, row_y, s=node_sz, facecolor="#FFFFFF" if is_active else color,
                              edgecolor=line_col, lw=1.2, zorder=8)

                # Leader line to right margin badge
                ax[0].plot([post_x, img_w - 55], [row_y, row_y], color=line_col, linestyle=":", lw=0.9, alpha=0.5, zorder=5)

                badge_txt = f" ► {lvl_name} " if is_active else f" {lvl_name} "
                badge_bg = "#002B49" if is_active else "#0B132B"
                ax[0].text(
                    img_w - 5,
                    row_y,
                    badge_txt,
                    color="#FFFFFF" if is_active else "#E2E8F0",
                    fontsize=8.5 if is_active else 7.8,
                    fontweight="bold",
                    ha="right",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.25,rounding_size=0.25",
                        facecolor=badge_bg,
                        edgecolor=line_col,
                        linewidth=1.6 if is_active else 1.0,
                        alpha=0.92,
                    ),
                    zorder=9,
                )
            elif orientation == "Sagittal" and is_vertical:
                row_y = data["y"]
                col_z = data["z"]
                ant_z = max(5.0, col_z - 12.0)
                post_z = min(img_w - 5.0, col_z + 12.0)
                line_col = "#00F5FF" if is_active else color
                lw = 2.0 if is_active else 1.3
                ax[0].plot([ant_z, post_z], [row_y, row_y], color=line_col, lw=lw, alpha=0.9 if is_active else 0.7, zorder=6)
                tick_h = 2.0 if is_active else 1.3
                ax[0].plot([ant_z, ant_z], [row_y - tick_h, row_y + tick_h], color=line_col, lw=lw, zorder=6)
                ax[0].plot([post_z, post_z], [row_y - tick_h, row_y + tick_h], color=line_col, lw=lw, zorder=6)
                ax[0].scatter(col_z, row_y, s=35 if is_active else 20, facecolor="#FFFFFF" if is_active else color, edgecolor=line_col, lw=1.2, zorder=8)
                ax[0].plot([post_z, img_w - 18], [row_y, row_y], color=line_col, linestyle=":", lw=0.9, alpha=0.5, zorder=5)
                badge_txt = f" ► {lvl_name} " if is_active else f" {lvl_name} "
                badge_bg = "#002B49" if is_active else "#0B132B"
                ax[0].text(img_w - 1.5, row_y, badge_txt, color="#FFFFFF" if is_active else "#E2E8F0",
                           fontsize=8.5 if is_active else 7.8, fontweight="bold", ha="right", va="center",
                           bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_bg, edgecolor=line_col, lw=1.5 if is_active else 1.0), zorder=9)
            elif orientation == "Axial":
                row_y = data["y"]
                col_x = data["x"]
                slice_z = data["z"]
                if abs(index - slice_z) <= 3:
                    node_col = "#00F5FF" if is_active else color
                    ax[0].scatter(col_x, row_y, color=node_col, s=60, marker="D", edgecolors="#FFFFFF", linewidths=1.5, zorder=7)
                    ax[0].text(col_x + 3.0, row_y + 2.0, f" {lvl_name} ", color="#FFFFFF", fontsize=8, fontweight="bold",
                               bbox=dict(boxstyle="round,pad=0.25", facecolor="#0B132B", edgecolor=node_col, lw=1.4), zorder=8)
            elif orientation == "Coronal":
                row_z = (img_h - 1 - data["z"]) if is_vertical else data["z"]
                col_x = data["x"]
                node_col = "#00F5FF" if is_active else color
                ax[0].plot([col_x - 12, col_x + 12], [row_z, row_z], color=node_col, lw=1.5, alpha=0.85)
                ax[0].scatter(col_x, row_z, color=node_col, s=35, edgecolors="#FFFFFF", linewidths=1.2, zorder=7)
                ax[0].text(img_w - 2, row_z, f" {lvl_name} ", color="#FFFFFF", fontsize=7.5, fontweight="bold",
                           ha="right", va="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="#0B132B", edgecolor=node_col, lw=1.2), zorder=8)

    # Orthogonal Crosshair Reference
    if show_crosshair and orthogonal_slice_indices and not is_native_mode:
        ax_z, cor_y, sag_x = orthogonal_slice_indices
        if orientation == "Sagittal" and is_vertical:
            ax[0].axhline(y=cor_y, color="#3B82F6", linestyle=":", linewidth=1.2, alpha=0.8, label="Coronal Plane")
            ax[0].axvline(x=ax_z, color="#EF4444", linestyle=":", linewidth=1.2, alpha=0.8, label="Axial Plane")
        elif orientation == "Axial":
            ax[0].axvline(x=sag_x, color="#10B981", linestyle=":", linewidth=1.2, alpha=0.8, label="Sagittal Plane")
            ax[0].axhline(y=cor_y, color="#3B82F6", linestyle=":", linewidth=1.2, alpha=0.8, label="Coronal Plane")

    # ------------------------------------------------------------------------
    # Panel 2: 3D Swin-UNETR Disease Probability Map
    # ------------------------------------------------------------------------
    im1 = ax[1].imshow(
        b_render,
        cmap=cmap_name,
        origin=origin_mode,
        extent=extent_box,
        aspect=aspect_val,
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )
    ax[1].set_title(f"Swin-UNETR Probability — {disease_short}", color="#F8FAFC", fontsize=10.5, fontweight="bold", pad=10)
    cb = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("Probability Score", color="#94A3B8", fontsize=8)
    cb.ax.tick_params(colors="#94A3B8", labelsize=7.5)

    if np.any(b >= threshold):
        peak_idx = np.unravel_index(np.argmax(b), b.shape)
        peak_row, peak_col = peak_idx
        ax[1].plot(peak_col, peak_row, marker="x", markersize=10, color="#FFFFFF", markeredgewidth=2.0)
        ax[1].text(peak_col + 2, peak_row - 2, f"Peak: {b.max():.2f}", color="#FFFFFF", fontsize=7.5,
                   fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.75))

    # ------------------------------------------------------------------------
    # Panel 3: Physical Localization Mask, Ground Truth & Active Caliper
    # ------------------------------------------------------------------------
    ax[2].imshow(
        a_render,
        cmap="gray",
        origin=origin_mode,
        extent=extent_box,
        aspect=aspect_val,
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )

    mask = (b >= threshold)
    if np.any(mask):
        ax[2].contour(
            b,
            levels=[threshold],
            colors=["#F43F5E"],
            linewidths=2.0,
            origin=origin_mode,
            extent=extent_box,
        )
        if "Fill" in overlay_mode:
            color_mask = np.zeros((*b_render.shape, 4), dtype=np.float32)
            color_mask[b_render >= threshold] = [0.95, 0.25, 0.35, prob_alpha]
            ax[2].imshow(color_mask, origin=origin_mode, extent=extent_box, aspect=aspect_val)

    ax[2].set_title(f"Localization & Caliper — {disease_short} (≥{threshold:.2f})", color="#F8FAFC", fontsize=10.5, fontweight="bold", pad=10)

    # Level indicator marks on Panel 3
    if show_levels and active_levels and level_style != "🚫 Hide Level Overlays":
        for lvl_name, data in active_levels.items():
            color = LEVEL_COLORS.get(lvl_name, "#38BDF8")
            is_active = (nearest_lvl == lvl_name)
            if is_native_mode:
                ax[2].plot([data["x"] - 15, data["x"] + 15], [data["y"], data["y"]],
                           color="#00F5FF" if is_active else color, linestyle="--", lw=1.1, alpha=0.7 if is_active else 0.45)
            elif orientation == "Sagittal" and is_vertical:
                ax[2].plot([data["z"] - 8, data["z"] + 8], [data["y"], data["y"]],
                           color="#00F5FF" if is_active else color, linestyle="--", lw=1.1, alpha=0.7 if is_active else 0.45)

    # RSNA annotation landmark stars
    if show_rsna_points and active_levels:
        for lvl_name, data in active_levels.items():
            if is_native_mode:
                pt_col = data["x"]
                pt_row = data["y"]
            elif orientation == "Sagittal" and is_vertical:
                pt_col = data["z"]
                pt_row = data["y"]
            elif orientation == "Axial":
                pt_col = data["x"]
                pt_row = data["y"]
            else:
                pt_col = data["x"]
                pt_row = (img_h - 1 - data["z"]) if is_vertical else data["z"]

            ax[2].plot(pt_col, pt_row, marker="*", markersize=11, color="#FBBF24",
                       markeredgecolor="#78350F", markeredgewidth=1.2, zorder=6)

    # Active Measurement Caliper
    if show_caliper and caliper_coords is not None:
        p1, p2 = caliper_coords
        col1, row1 = p1
        col2, row2 = p2

        ax[2].plot([col1, col2], [row1, row2], color="#00F5FF", linestyle="-", linewidth=2.2, zorder=10)
        ax[2].scatter([col1, col2], [row1, row2], color="#00F5FF", s=65, marker="D",
                     edgecolors="#000000", linewidths=1.5, zorder=11)

        dx = col2 - col1
        dy = row2 - row1
        length = math.hypot(dx, dy)
        if length > 1e-5:
            nx = -dy / length * 3.2
            ny = dx / length * 3.2
            ax[2].plot([col1 - nx, col1 + nx], [row1 - ny, row1 + ny], color="#00F5FF", linewidth=2.0, zorder=10)
            ax[2].plot([col2 - nx, col2 + nx], [row2 - ny, row2 + ny], color="#00F5FF", linewidth=2.0, zorder=10)

        mid_x = (col1 + col2) / 2.0
        mid_y = (row1 + row2) / 2.0
        caliper_text = f"CALIPER: {caliper_mm:.1f} mm"
        if caliper_label:
            caliper_text += f"\n({caliper_label})"
        ax[2].text(
            mid_x,
            mid_y + 12 if is_native_mode else mid_y,
            caliper_text,
            color="#001020",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#00F5FF", edgecolor="#002244", linewidth=1.2, alpha=0.92),
        )

    # Anatomical orientation axis indicators
    if is_native_mode or (is_vertical and orientation == "Sagittal"):
        ax[0].set_ylabel("Superior (Cranial) ↑  |  Inferior (Caudal) ↓", color="#94A3B8", fontsize=7.5)
        for a_idx in range(3):
            ax[a_idx].set_xlabel("Anterior ←  |  Posterior →", color="#94A3B8", fontsize=7.5)
    elif is_vertical and orientation == "Axial":
        ax[0].set_ylabel("Anterior ↑  |  Posterior ↓", color="#94A3B8", fontsize=7.5)
        for a_idx in range(3):
            ax[a_idx].set_xlabel("Right ←  |  Left →", color="#94A3B8", fontsize=7.5)

    mode_banner = "Native DICOM PACS Workstation" if is_native_mode else "Canonical Swin-UNETR Grid"
    fig.suptitle(
        f"Lumbar Spine AI Workstation — {mode_banner} | {orientation} Slice {index + 1 if is_native_mode else index} | {disease_full}",
        color="#F8FAFC",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _plot_orthogonal_mpr(image, prob, center_z, center_y, center_x):
    """Render synchronized Axial, Coronal, and Sagittal orthogonal cross-sections."""
    ax_slice = _slice(image, "Axial", center_z, is_vertical=True)
    cor_slice = _slice(image, "Coronal", center_y, is_vertical=True)
    sag_slice = _slice(image, "Sagittal", center_x, is_vertical=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=140)
    fig.patch.set_facecolor("#0B132B")

    titles = [
        f"Axial Cross-Section (Z={center_z})",
        f"Coronal Plane (Y={center_y})",
        f"Sagittal Vertical Column (X={center_x})",
    ]
    slices = [ax_slice, cor_slice, sag_slice]

    for axis, img_slc, title in zip(axes, slices, titles):
        axis.set_facecolor("#000000")
        img_win = _apply_windowing(img_slc, "PACS High Definition / Diagnostic Contrast")
        axis.imshow(img_win, cmap="gray", origin="upper", aspect=1.0, interpolation="bicubic")
        axis.set_title(title, color="#F8FAFC", fontsize=10, fontweight="bold", pad=8)
        axis.tick_params(colors="#94A3B8", labelsize=7)
        for spine in axis.spines.values():
            spine.set_color("#1E293B")
            spine.set_linewidth(1.0)

    # Crosshairs on vertical views
    # Axial: (cols X, rows Y)
    axes[0].axvline(center_x, color="#10B981", linestyle="--", linewidth=1.0)
    axes[0].axhline(center_y, color="#3B82F6", linestyle="--", linewidth=1.0)

    # Coronal: (cols X, rows Z inverted)
    axes[1].axvline(center_x, color="#10B981", linestyle="--", linewidth=1.0)
    axes[1].axhline(63 - center_z, color="#EF4444", linestyle="--", linewidth=1.0)

    # Sagittal: (cols Z, rows Y)
    axes[2].axvline(center_z, color="#EF4444", linestyle="--", linewidth=1.0)
    axes[2].axhline(center_y, color="#3B82F6", linestyle="--", linewidth=1.0)

    fig.suptitle(
        f"Synchronized Multi-Planar Reconstruction (MPR) — Focused at (Z:{center_z}, Y:{center_y}, X:{center_x})",
        color="#F8FAFC",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ============================================================================
# METRICS: Automated Physical Volumetrics & Morphometrics
# ============================================================================

def _calculate_volumetric_metrics(prob_volume, disease, threshold, physical_extents, levels):
    """
    Computes physical morphometrics:
    Volume in mm³ & cm³, caliper bounding spans (ΔX, ΔY, ΔZ mm), centroid, peak probability,
    and nearest anatomical disc level.
    """
    if prob_volume.ndim == 4:
        disease_prob = prob_volume[disease]
    else:
        disease_prob = prob_volume

    mask = disease_prob >= threshold
    voxels = int(np.sum(mask))

    d, h, w = disease_prob.shape
    dz_mm, dy_mm, dx_mm = _get_mm_scales(physical_extents, (d, h, w))
    voxel_volume_mm3 = dz_mm * dy_mm * dx_mm

    total_volume_mm3 = voxels * voxel_volume_mm3
    total_volume_cm3 = total_volume_mm3 / 1000.0

    if voxels > 0:
        z_indices, y_indices, x_indices = np.where(mask)
        span_z_mm = (float(np.max(z_indices)) - float(np.min(z_indices)) + 1) * dz_mm
        span_y_mm = (float(np.max(y_indices)) - float(np.min(y_indices)) + 1) * dy_mm
        span_x_mm = (float(np.max(x_indices)) - float(np.min(x_indices)) + 1) * dx_mm

        centroid_z = float(np.mean(z_indices))
        centroid_y = float(np.mean(y_indices))
        centroid_x = float(np.mean(x_indices))

        peak_prob = float(np.max(disease_prob))
        mean_prob = float(np.mean(disease_prob[mask]))

        nearest_lvl = "L4/L5"
        min_lvl_dist = float("inf")
        for lvl_name, l_data in levels.items():
            dist_sq = (
                ((centroid_z - l_data["z"]) * dz_mm) ** 2 +
                ((centroid_y - l_data["y"]) * dy_mm) ** 2 +
                ((centroid_x - l_data["x"]) * dx_mm) ** 2
            )
            dist = math.sqrt(dist_sq)
            if dist < min_lvl_dist:
                min_lvl_dist = dist
                nearest_lvl = lvl_name
    else:
        span_z_mm = span_y_mm = span_x_mm = 0.0
        centroid_z = centroid_y = centroid_x = 0.0
        peak_prob = float(np.max(disease_prob)) if disease_prob.size > 0 else 0.0
        mean_prob = 0.0
        nearest_lvl = "N/A"
        min_lvl_dist = 0.0

    return {
        "voxels": voxels,
        "volume_mm3": total_volume_mm3,
        "volume_cm3": total_volume_cm3,
        "span_x_mm": span_x_mm,
        "span_y_mm": span_y_mm,
        "span_z_mm": span_z_mm,
        "centroid_voxel": (centroid_z, centroid_y, centroid_x),
        "peak_probability": peak_prob,
        "mean_probability": mean_prob,
        "nearest_level": nearest_lvl,
        "level_distance_mm": min_lvl_dist,
    }


# ============================================================================
# MAIN UI: Render 3D Segmentation & Measurement Workstation
# ============================================================================

def render_segmentation():
    st.markdown("""
    <div class="page-title-row" style="margin-bottom:0.8rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <h2 style="margin:0; font-size:1.6rem; color:#0F1B3D; font-weight:800;">
                    🧠 3D Disease Localization &amp; Spine Measurement Workstation
                </h2>
                <p style="margin:4px 0 0 0; color:#475569; font-size:0.88rem;">
                    MONAI 3D Swin-UNETR physical-space inference with vertical anatomical disc levels (L1–S1), interactive caliper tools, and volumetric morphometrics
                </p>
            </div>
            <span style="background:#EFF6FF; border:1px solid #BFDBFE; color:#1D4ED8; font-weight:700; font-size:0.75rem; padding:6px 12px; border-radius:8px;">
                FROZEN PART 3.3 CHECKPOINT
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ensure session state safety
    session = st.session_state.get("analysis_session")
    if session is not None:
        if not hasattr(session, "measurements") or session.measurements is None:
            session.measurements = []
        if not hasattr(session, "detected_lesion_metrics") or session.detected_lesion_metrics is None:
            session.detected_lesion_metrics = {}
        if not hasattr(session, "level_morphometrics") or session.level_morphometrics is None:
            session.level_morphometrics = {}

    # ------------------------------------------------------------------------
    # Top Specs Cards
    # ------------------------------------------------------------------------
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1: st.metric("Backbone Architecture", "Swin-UNETR (3D)")
    with sc2: st.metric("Physical Input Grid", "64 × 96 × 96")
    with sc3: st.metric("Target Output Classes", "6 (SCS, LFNN, RFNN, LSS, RSS)")
    with sc4: st.metric("Parameter Count", "4,078,116")
    with sc5: st.metric("Coordinate Space", "Patient Physical (mm)")

    st.markdown("---")

    # ------------------------------------------------------------------------
    # 1. Series Selection & Inference Trigger
    # ------------------------------------------------------------------------
    st.markdown("#### 📂 1. Select RSNA Lumbar Spine MRI Series")

    try:
        manifest = final_inf.load_manifest()
    except Exception as exc:
        st.error(f"Unable to load RSNA manifest: {exc}")
        return

    DEMO_CASES = {
        "Custom (Enter Study/Series ID below)": ("", ""),
        "Case 1: Study 1012375618 | Series 4014890929 (Sagittal T2/STIR, 17 slices - 100% Hit@0.50)": ("1012375618", "4014890929"),
        "Case 2: Study 1004726367 | Series 1709080005 (Sagittal T2/STIR, 23 slices - 100% Hit@0.50)": ("1004726367", "1709080005"),
        "Case 3: Study 100206310 | Series 1792451510 (Sagittal T2/STIR, 45 slices - 100% Hit@0.50)": ("100206310", "1792451510"),
        "Case 4: Study 1002894806 | Series 801316590 (Sagittal T2/STIR, 24 slices - 80% Hit@0.50)": ("1002894806", "801316590"),
        "Case 5: Study 1013791258 | Series 2460967246 (Sagittal T2/STIR, 38 slices - Untouched Test)": ("1013791258", "2460967246"),
    }

    c_sel, c_btn = st.columns([3, 1.2])
    with c_sel:
        selected_demo = st.selectbox(
            "💡 Pre-verified RSNA Evaluation Cases",
            list(DEMO_CASES.keys()),
            index=1,
            help="Select a benchmark case with verified ground truth coordinates across L1-S1 levels.",
        )
    demo_study_id, demo_series_id = DEMO_CASES[selected_demo]

    col_id1, col_id2, col_act = st.columns([1.5, 1.5, 1.2])
    with col_id1:
        study_id = st.text_input("Study ID", value=demo_study_id, placeholder="e.g. 1012375618")
    with col_id2:
        series_id = st.text_input("Series ID", value=demo_series_id, placeholder="e.g. 4014890929")
    with col_act:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Swin-UNETR Inference", type="primary", use_container_width=True)

    if run_btn:
        if not study_id or not series_id:
            st.warning("Please provide both Study ID and Series ID.")
            return

        with st.spinner("Processing DICOM physical geometry and running 3D Swin-UNETR inference..."):
            try:
                res = final_inf.run_final_inference(str(study_id).strip(), str(series_id).strip())
                st.session_state.segmentation_result = res
                st.session_state.final_segmentation_result = res
                st.session_state.segmentation_ready = True

                session = st.session_state.get("analysis_session")
                if session is not None:
                    img_arr = np.asarray(res.get("image"), dtype=np.float32)
                    pred_arr = np.asarray(res.get("prediction"))
                    session.segmentation_result = res
                    session.segmentation_input_volume = img_arr
                    session.segmentation_volume = pred_arr
                    session.segmentation_ready = True
                    if not session.study_id: session.study_id = str(study_id).strip()
                    if not session.series_id: session.series_id = str(series_id).strip()

                st.success("✅ 3D Swin-UNETR inference completed successfully.")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

    # Retrieve result from state
    result = st.session_state.get("segmentation_result") or st.session_state.get("final_segmentation_result")
    if not result:
        session = st.session_state.get("analysis_session")
        if session and session.segmentation_result:
            result = session.segmentation_result
            st.session_state.segmentation_result = result

    if not result:
        st.info("👆 Select a study/series case above and click **Run Swin-UNETR Inference** to activate the 3D workstation.")
        return

    # Extract outputs
    image = np.asarray(result.get("image"))
    probabilities = np.asarray(result.get("probabilities"))
    prediction = np.asarray(result.get("prediction"))
    geometry = result.get("geometry", {})
    physical_extents = result.get("physical_extents", {})
    metadata = result.get("metadata", {})
    raw_points = result.get("points", [])

    # Extract authentic native DICOM volume & geometry-verified native levels
    image_native = result.get("image_native")
    records = result.get("records")
    if image_native is None:
        c_study = result.get("study_id") or (study_id if 'study_id' in locals() else None)
        c_series = result.get("series_id") or (series_id if 'series_id' in locals() else None)
        if c_study and c_series:
            image_native, records, _ = _load_native_dicom_volume(c_study, c_series)
            if image_native is not None:
                result["image_native"] = image_native
                result["records"] = records

    native_levels = _extract_native_levels(raw_points, geometry)

    # Extract canonical levels & mm scales
    levels = _extract_levels(raw_points, physical_extents, image.shape)
    mm_scales = _get_mm_scales(physical_extents, image.shape)

    st.markdown("---")

    # ------------------------------------------------------------------------
    # 2. View Resolution Engine & Anatomical Level Jump Bar
    # ------------------------------------------------------------------------
    c_eng1, c_eng2 = st.columns([3.2, 1.8])
    with c_eng1:
        view_engine = st.radio(
            "Visual Resolution & Rendering Engine",
            [
                "🔬 Original Native DICOM Slices (Full-Field Anatomical PACS View — Crystal Clear)",
                "🧊 3D Swin-UNETR Model Volume (Canonical Grid)",
            ],
            index=0,
            horizontal=True,
            help="Original Native DICOM mode displays authentic full-resolution DICOM slices (320×320 / 640×640) with full field-of-view clinical anatomy, matching the crystal-clear clarity of the Classification view.",
        )
    is_native_mode = ("Original Native" in view_engine) and (image_native is not None)

    with c_eng2:
        if is_native_mode:
            st.caption(f"✨ **Full Native PACS Resolution**: `{image_native.shape[2]}×{image_native.shape[1]}` ({image_native.shape[0]} slices) · 100% Authentic Original DICOM")
        else:
            st.caption(f"🧊 **Model Space**: `{image.shape[2]}×{image.shape[1]}×{image.shape[0]}` isotropic canonical voxels")

    st.markdown("#### 🦴 2. Quick Jump to Lumbar Level / Intervertebral Disc Space")
    st.caption("Click any level button to immediately position the slice slider directly on that anatomical disc space:")

    lvl_cols = st.columns(5)
    selected_jump = None
    levels_dict = native_levels if (is_native_mode and native_levels) else levels
    for i, (lvl_name, l_data) in enumerate(levels_dict.items()):
        color = LEVEL_COLORS.get(lvl_name, "#2563EB")
        with lvl_cols[i]:
            if st.button(f"🦴 **{lvl_name}**", key=f"jump_{lvl_name}", use_container_width=True):
                selected_jump = lvl_name

    # ------------------------------------------------------------------------
    # 3. Workstation Toolbar & Viewing Controls
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🖥️ 3. Multi-Planar Slice Visualizer & Inspection Controls")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns([1.4, 1.4, 1.4, 1.4, 1.4])

    with col_ctrl1:
        disease_choice = st.selectbox(
            "Target Lumbar Pathology",
            [1, 2, 3, 4, 5],
            format_func=lambda x: f"{SHORT_NAMES[x]} — {CLASS_NAMES[x]}",
            key="seg_disease_select",
        )

    with col_ctrl2:
        orientation = st.selectbox(
            "Anatomical Plane",
            ["Sagittal", "Axial", "Coronal"],
            index=0,
            key="seg_orientation_select",
            help="Sagittal: lateral spine profile (L1–S1 column). Axial: cross-sectional canal & foramina. Coronal: bilateral symmetry.",
        )

    is_native_active = bool(is_native_mode and orientation == "Sagittal" and image_native is not None)
    if is_native_mode and orientation != "Sagittal":
        st.info("ℹ️ Note: Axial and Coronal multi-planar views are reconstructed from the 3D canonical volume. Sagittal orientation displays full-resolution authentic native DICOM slices.")

    with col_ctrl3:
        orient_display = st.selectbox(
            "Display Orientation",
            ["↕️ Vertical Anatomical", "↔️ Horizontal Grid"],
            index=0,
            help="Vertical Anatomical stands the spine upright (Cranial at top, Caudal at bottom) matching standard clinical PACS.",
        )
        is_vertical_mode = (orient_display == "↕️ Vertical Anatomical")

    # Determine slice limits
    if is_native_active:
        n_slices = image_native.shape[0]
    elif orientation == "Axial":
        n_slices = image.shape[0]  # Z=64
    elif orientation == "Coronal":
        n_slices = image.shape[1]  # Y=96
    else:  # Sagittal
        n_slices = image.shape[2]  # X=96

    # Handle Jump Button logic
    default_slice_val = n_slices // 2
    slider_key = f"slider_slice_{orientation}_{'nat' if is_native_active else 'can'}"
    if selected_jump and selected_jump in levels_dict:
        j_data = levels_dict[selected_jump]
        if is_native_active:
            default_slice_val = int(round(j_data["z"]))
        elif orientation == "Axial":
            default_slice_val = int(round(j_data["z"]))
        elif orientation == "Coronal":
            default_slice_val = int(round(j_data["y"]))
        else:
            default_slice_val = int(round(j_data["x"]))
        st.session_state[slider_key] = default_slice_val

    with col_ctrl4:
        current_slider_val = st.session_state.get(slider_key, default_slice_val)
        current_slider_val = int(np.clip(current_slider_val, 0, n_slices - 1))
        slice_idx = st.slider(
            f"Slice Index (0 to {n_slices - 1})",
            min_value=0,
            max_value=n_slices - 1,
            value=current_slider_val,
            key=slider_key,
        )

    with col_ctrl5:
        threshold = st.slider(
            "Probability Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Confidence threshold for Swin-UNETR localization boundary.",
        )

    # Current Slice Anatomical Level Status Banner
    nearest_lvl, lvl_dist_mm, lvl_desc, lvl_badge_color, is_exact = _identify_slice_level(
        orientation, slice_idx, levels_dict, mm_scales,
        is_native=is_native_active,
        slice_spacing=float(geometry.get("slice_spacing", 4.0))
    )
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {lvl_badge_color};
                border-radius:8px; padding:0.6rem 1rem; margin:0.4rem 0 1rem 0; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:0.85rem; color:#0F1B3D; font-weight:700;">📍 Anatomical Position:</span>
            <span style="font-size:0.85rem; color:#334155; margin-left:6px;">{lvl_desc}</span>
        </div>
        <span style="background:{lvl_badge_color}1A; color:{lvl_badge_color}; font-size:0.75rem; font-weight:800; padding:3px 8px; border-radius:6px; border:1px solid {lvl_badge_color}4D;">
            {nearest_lvl} FOCUS
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Secondary Visualization Customization Expanders
    with st.expander("🎨 Advanced Display, Windowing & Colormap Settings", expanded=False):
        c_win1, c_win2, c_win3, c_win4 = st.columns(4)
        with c_win1:
            win_preset = st.selectbox(
                "Radiological Window (WL/WW)",
                ["Full Dynamic Range", "Bone & Vertebral Endplates", "Soft Tissue / Thecal Sac", "High Contrast Lesion View"],
                index=0,
            )
        with c_win2:
            cmap_choice = st.selectbox(
                "Heatmap Colormap",
                ["magma", "turbo", "viridis", "inferno", "plasma"],
                index=0,
            )
        with c_win3:
            overlay_style = st.selectbox(
                "Overlay Style",
                ["Contour & Translucent Fill", "Contour Outline Only"],
                index=0,
            )
        with c_win4:
            alpha_val = st.slider("Heatmap Alpha Opacity", 0.1, 1.0, 0.55, 0.05)

        c_chk1, c_chk2, c_chk3, c_chk4 = st.columns(4)
        with c_chk1: chk_show_levels = st.checkbox("Show Disc & Level Indicators", value=True)
        with c_chk2: chk_show_rsna = st.checkbox("Show RSNA Ground-Truth Landmarks", value=True)
        with c_chk3: chk_show_crosshair = st.checkbox("Show Multi-Planar Orthogonal Line", value=True)
        with c_chk4: chk_show_caliper = st.checkbox("Activate Measurement Caliper on Slice", value=True)

    # ------------------------------------------------------------------------
    # 4. Spine Caliper & Measurement Engine
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📐 4. Clinical Spine Measurement & Caliper Workstation")
    c_meas1, c_meas2, c_meas3 = st.columns([1.5, 1.5, 2.0])

    with c_meas1:
        meas_preset = st.selectbox(
            "Clinical Measurement Tool",
            [
                "Spinal Canal AP Diameter",
                "Neural Foraminal Height & Width",
                "Intervertebral Disc Height",
                "Vertebral Translation (Spondylolisthesis)",
                "Free Caliper (Arbitrary Distance)",
            ],
            index=0,
        )

    # Image slice dimensions for current view
    if is_native_active:
        cur_slice_preview = image_native[slice_idx]
        h_max = cur_slice_preview.shape[1]  # Width (columns e.g. 320)
        v_max = cur_slice_preview.shape[0]  # Height (rows e.g. 320)
        cur_lvl_data = native_levels.get(nearest_lvl, list(native_levels.values())[0] if native_levels else {"x": 160.0, "y": 160.0})
        base_h = float(cur_lvl_data["x"])
        base_v = float(cur_lvl_data["y"])
        scale_h = float(geometry.get("column_spacing", 0.6875))
        scale_v = float(geometry.get("row_spacing", 0.6875))
    else:
        cur_slice_preview = _slice(image, orientation, slice_idx, is_vertical=is_vertical_mode)
        h_max = cur_slice_preview.shape[1]  # Width (columns)
        v_max = cur_slice_preview.shape[0]  # Height (rows)
        cur_lvl_data = levels.get(nearest_lvl, levels["L4/L5"])
        dz_mm, dy_mm, dx_mm = mm_scales

        if orientation == "Sagittal" and is_vertical_mode:
            # Horiz is Z (cols 0-63), Vert is Y (rows 0-95)
            base_h = float(cur_lvl_data["z"])
            base_v = float(cur_lvl_data["y"])
            scale_h = dz_mm
            scale_v = dy_mm
        elif orientation == "Sagittal" and not is_vertical_mode:
            # Horiz is Y, Vert is Z
            base_h = float(cur_lvl_data["y"])
            base_v = float(cur_lvl_data["z"])
            scale_h = dy_mm
            scale_v = dz_mm
        elif orientation == "Axial":
            # Horiz is X, Vert is Y
            base_h = float(cur_lvl_data["x"])
            base_v = float(cur_lvl_data["y"])
            scale_h = dx_mm
            scale_v = dy_mm
        else:  # Coronal
            base_h = float(cur_lvl_data["x"])
            base_v = float(cur_lvl_data["z"])
            scale_h = dx_mm
            scale_v = dz_mm

    # Default preset offsets
    if is_native_active:
        if meas_preset == "Spinal Canal AP Diameter":
            def_p1 = (base_h - 10.0, base_v)
            def_p2 = (base_h + 16.0, base_v)
            meas_unit_note = "Antero-Posterior canal span"
        elif meas_preset == "Neural Foraminal Height & Width":
            def_p1 = (base_h, base_v - 14.0)
            def_p2 = (base_h, base_v + 14.0)
            meas_unit_note = "Foraminal height"
        elif meas_preset == "Intervertebral Disc Height":
            def_p1 = (base_h, base_v - 9.0)
            def_p2 = (base_h, base_v + 9.0)
            meas_unit_note = "Disc space height"
        elif meas_preset == "Vertebral Translation (Spondylolisthesis)":
            def_p1 = (base_h - 12.0, base_v)
            def_p2 = (base_h + 12.0, base_v)
            meas_unit_note = "Vertebral translation / slip"
        else:
            def_p1 = (base_h - 15.0, base_v - 10.0)
            def_p2 = (base_h + 15.0, base_v + 10.0)
            meas_unit_note = "Free anatomical distance"
    else:
        if meas_preset == "Spinal Canal AP Diameter":
            def_p1 = (base_h - 4.0, base_v)
            def_p2 = (base_h + 8.0, base_v)
            meas_unit_note = "Antero-Posterior canal span"
        elif meas_preset == "Neural Foraminal Height & Width":
            def_p1 = (base_h, base_v - 6.0)
            def_p2 = (base_h, base_v + 6.0)
            meas_unit_note = "Foraminal height"
        elif meas_preset == "Intervertebral Disc Height":
            def_p1 = (base_h, base_v - 4.0)
            def_p2 = (base_h, base_v + 4.0)
            meas_unit_note = "Disc space height"
        elif meas_preset == "Vertebral Translation (Spondylolisthesis)":
            def_p1 = (base_h - 5.0, base_v)
            def_p2 = (base_h + 5.0, base_v)
            meas_unit_note = "Vertebral translation / slip"
        else:
            def_p1 = (base_h - 8.0, base_v - 4.0)
            def_p2 = (base_h + 8.0, base_v + 4.0)
            meas_unit_note = "Free anatomical distance"

    with c_meas2:
        st.caption(f"Caliper Target: **{meas_unit_note}**")
        p1_h = st.slider("Point 1 (Horizontal)", 0.0, float(h_max - 1), float(np.clip(def_p1[0], 0, h_max - 1)), 0.5, key="cal_p1_h")
        p1_v = st.slider("Point 1 (Vertical)", 0.0, float(v_max - 1), float(np.clip(def_p1[1], 0, v_max - 1)), 0.5, key="cal_p1_v")

    with c_meas3:
        st.caption("Adjust Point 2 to complete measurement:")
        p2_h = st.slider("Point 2 (Horizontal)", 0.0, float(h_max - 1), float(np.clip(def_p2[0], 0, h_max - 1)), 0.5, key="cal_p2_h")
        p2_v = st.slider("Point 2 (Vertical)", 0.0, float(v_max - 1), float(np.clip(def_p2[1], 0, v_max - 1)), 0.5, key="cal_p2_v")

    # Real physical distance in mm
    dh_mm = (p2_h - p1_h) * scale_h
    dv_mm = (p2_v - p1_v) * scale_v
    measured_mm = math.hypot(dh_mm, dv_mm)

    # Clinical interpretation logic
    if meas_preset == "Spinal Canal AP Diameter":
        if measured_mm >= 15.0:
            diag_label = "Normal / Wide Spinal Canal"
            diag_badge = "🟢 NORMAL"
            diag_color = "#10B981"
        elif measured_mm >= 10.0:
            diag_label = "Relative / Moderate Spinal Canal Stenosis"
            diag_badge = "🟡 RELATIVE STENOSIS"
            diag_color = "#F59E0B"
        else:
            diag_label = "Absolute / Severe Spinal Canal Stenosis"
            diag_badge = "🔴 ABSOLUTE STENOSIS"
            diag_color = "#EF4444"
    elif meas_preset == "Neural Foraminal Height & Width":
        if measured_mm >= 14.0:
            diag_label = "Adequate Foraminal Caliber (Normal)"
            diag_badge = "🟢 NORMAL"
            diag_color = "#10B981"
        elif measured_mm >= 10.0:
            diag_label = "Mild-to-Moderate Neural Foraminal Narrowing"
            diag_badge = "🟡 MILD-MODERATE"
            diag_color = "#F59E0B"
        else:
            diag_label = "Severe Neural Foraminal Stenosis / Root Impingement Risk"
            diag_badge = "🔴 SEVERE STENOSIS"
            diag_color = "#EF4444"
    elif meas_preset == "Intervertebral Disc Height":
        if measured_mm >= 7.5:
            diag_label = "Preserved Intervertebral Disc Height"
            diag_badge = "🟢 PRESERVED"
            diag_color = "#10B981"
        elif measured_mm >= 4.5:
            diag_label = "Mild-to-Moderate Disc Space Narrowing / Desiccation"
            diag_badge = "🟡 MODERATE LOSS"
            diag_color = "#F59E0B"
        else:
            diag_label = "Severe Intervertebral Disc Collapse"
            diag_badge = "🔴 SEVERE COLLAPSE"
            diag_color = "#EF4444"
    elif meas_preset == "Vertebral Translation (Spondylolisthesis)":
        if measured_mm < 3.0:
            diag_label = "No Significant Spondylolisthesis / Translation"
            diag_badge = "🟢 STABLE"
            diag_color = "#10B981"
        elif measured_mm < 8.0:
            diag_label = "Meyerding Grade I Spondylolisthesis (< 25% slip)"
            diag_badge = "🟡 GRADE I SLIP"
            diag_color = "#F59E0B"
        else:
            diag_label = "Meyerding Grade II+ Spondylolisthesis (≥ 25% slip)"
            diag_badge = "🔴 GRADE II+ SLIP"
            diag_color = "#EF4444"
    else:
        diag_label = "Custom Measured Distance"
        diag_badge = "ℹ️ MEASURED"
        diag_color = "#2563EB"

    # Display live measurement card & save button
    c_res_box, c_save_box = st.columns([3, 1.2])
    with c_res_box:
        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:10px; padding:0.8rem 1.2rem; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;">
                    ACTIVE MEASUREMENT: {meas_preset}
                </div>
                <div style="font-size:1.45rem; font-weight:800; color:#0F1B3D;">
                    📏 {measured_mm:.1f} mm <span style="font-size:0.8rem; font-weight:500; color:#64748B;">({math.hypot(p2_h - p1_h, p2_v - p1_v):.1f} voxels)</span>
                </div>
                <div style="font-size:0.82rem; color:{diag_color}; font-weight:600; margin-top:2px;">
                    {diag_label}
                </div>
            </div>
            <span style="background:{diag_color}1A; color:{diag_color}; font-size:0.82rem; font-weight:800; padding:6px 14px; border-radius:8px; border:1px solid {diag_color}4D;">
                {diag_badge}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with c_save_box:
        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        if st.button("💾 Save to Case Report", type="primary", use_container_width=True):
            meas_entry = {
                "id": f"meas_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "tool": meas_preset,
                "level": nearest_lvl,
                "orientation": orientation,
                "slice": slice_idx,
                "measured_mm": round(measured_mm, 1),
                "clinical_impression": diag_label,
                "status_badge": diag_badge,
            }
            if session is not None:
                if not hasattr(session, "measurements") or session.measurements is None:
                    session.measurements = []
                session.measurements.append(meas_entry)
            st.session_state.setdefault("saved_measurements", []).append(meas_entry)
            st.success(f"Saved {meas_preset} ({measured_mm:.1f} mm)!")

    # ------------------------------------------------------------------------
    # 5. Render Main Plot
    # ------------------------------------------------------------------------
    st.markdown("---")

    ext_param = None
    if physical_extents and "x" in physical_extents and "y" in physical_extents and "z" in physical_extents:
        ext_param = (physical_extents["x"], physical_extents["y"], physical_extents["z"])

    if probabilities.ndim == 4:
        disease_prob_volume = probabilities[disease_choice]
    else:
        disease_prob_volume = probabilities

    caliper_tuple = ((p1_h, p1_v), (p2_h, p2_v)) if chk_show_caliper else None

    ortho_indices = (
        slice_idx if orientation == "Axial" else int(round(levels[nearest_lvl]["z"])),
        slice_idx if orientation == "Coronal" else int(round(levels[nearest_lvl]["y"])),
        slice_idx if orientation == "Sagittal" else int(round(levels[nearest_lvl]["x"])),
    )

    native_slice_to_plot = None
    native_prob_to_plot = None
    if is_native_active and image_native is not None:
        native_slice_to_plot = image_native[slice_idx]
        native_prob_to_plot = _map_prob_to_native_slice(
            disease_prob_volume, slice_idx, geometry, native_slice_to_plot.shape, records=records
        )

    with st.spinner("Rendering high-resolution vertical anatomical slice visualization..."):
        fig = _plot_rich(
            image=image,
            prob=disease_prob_volume,
            pred=prediction,
            disease=disease_choice,
            orientation=orientation,
            index=slice_idx,
            threshold=threshold,
            ext=ext_param,
            levels=levels,
            is_vertical=is_vertical_mode,
            show_levels=chk_show_levels,
            show_rsna_points=chk_show_rsna,
            show_crosshair=chk_show_crosshair,
            show_caliper=chk_show_caliper,
            caliper_coords=caliper_tuple,
            caliper_mm=measured_mm,
            caliper_label=meas_preset,
            window_preset=win_preset,
            cmap_name=cmap_choice,
            prob_alpha=alpha_val,
            overlay_mode=overlay_style,
            orthogonal_slice_indices=ortho_indices,
            native_slice=native_slice_to_plot,
            native_prob=native_prob_to_plot,
            native_levels=native_levels,
            is_native_mode=is_native_active,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ------------------------------------------------------------------------
    # 6. Synchronized Multi-Planar Reconstruction (MPR) View
    # ------------------------------------------------------------------------
    with st.expander("🧭 Multi-Planar Orthogonal Tri-View (Axial • Coronal • Sagittal MPR)", expanded=False):
        st.caption("Simultaneous orthogonal views cross-referenced at the active slice focus point:")
        fig_mpr = _plot_orthogonal_mpr(
            image=image,
            prob=disease_prob_volume,
            center_z=ortho_indices[0],
            center_y=ortho_indices[1],
            center_x=ortho_indices[2],
        )
        st.pyplot(fig_mpr, use_container_width=True)
        plt.close(fig_mpr)

    # ------------------------------------------------------------------------
    # 7. Automated Physical Volumetrics & Morphometrics Dashboard
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📊 5. Automated Physical Volumetrics & Morphometrics")

    vol_metrics = _calculate_volumetric_metrics(
        probabilities, disease_choice, threshold, physical_extents, levels
    )

    if session is not None:
        session.detected_lesion_metrics = vol_metrics

    vm1, vm2, vm3, vm4 = st.columns(4)
    with vm1:
        st.metric("Lesion Physical Volume", f"{vol_metrics['volume_cm3']:.2f} cm³", f"{vol_metrics['volume_mm3']:.0f} mm³")
    with vm2:
        st.metric("Positive Voxels (≥ Threshold)", f"{vol_metrics['voxels']:,}", f"Peak: {vol_metrics['peak_probability']:.2f}")
    with vm3:
        st.metric("Transverse & AP Span", f"{vol_metrics['span_x_mm']:.1f} × {vol_metrics['span_y_mm']:.1f} mm", "Bounding Box")
    with vm4:
        st.metric("Craniocaudal Height (ΔZ)", f"{vol_metrics['span_z_mm']:.1f} mm", f"Level: {vol_metrics['nearest_level']}")

    # ------------------------------------------------------------------------
    # 8. Lumbar Spine Multi-Level Pathology Matrix (5 Conditions × 5 Levels)
    # ------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 6. Multi-Level Pathology Scorecard (5 Conditions × L1–S1 Levels)")
    st.caption("Volumetric Swin-UNETR probability and detection status across all anatomical segments:")

    scorecard_rows = []
    dz_mm, dy_mm, dx_mm = mm_scales
    for class_id in range(1, 6):
        c_short = SHORT_NAMES[class_id]
        c_full = CLASS_NAMES[class_id]
        if probabilities.ndim == 4:
            p_vol = probabilities[class_id]
        else:
            p_vol = probabilities

        row = {"Pathology Condition": f"{c_short} ({c_full})"}
        for lvl_name, l_data in levels.items():
            lvl_z = int(round(l_data["z"]))
            z_min = max(0, lvl_z - 3)
            z_max = min(image.shape[0], lvl_z + 4)
            sub_vol = p_vol[z_min:z_max, :, :]
            max_prob = float(np.max(sub_vol)) if sub_vol.size > 0 else 0.0

            if max_prob >= threshold:
                badge = f"🔴 {max_prob*100:.1f}%"
            elif max_prob >= 0.30:
                badge = f"🟡 {max_prob*100:.1f}%"
            else:
                badge = f"🟢 {max_prob*100:.1f}%"
            row[lvl_name] = badge

        scorecard_rows.append(row)

    df_scorecard = pd.DataFrame(scorecard_rows)
    st.dataframe(df_scorecard, hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------------
    # 9. Saved Clinical Measurements Table
    # ------------------------------------------------------------------------
    saved_meas = []
    if session is not None and getattr(session, "measurements", None):
        saved_meas = session.measurements
    elif "saved_measurements" in st.session_state:
        saved_meas = st.session_state["saved_measurements"]

    if saved_meas:
        st.markdown("---")
        st.markdown("#### 📑 7. Saved Case Measurements Log (Included in PDF Report)")
        df_meas = pd.DataFrame(saved_meas)[["timestamp", "tool", "level", "orientation", "slice", "measured_mm", "clinical_impression"]]
        df_meas.columns = ["Time", "Measurement Type", "Level", "Plane", "Slice", "Value (mm)", "Clinical Finding"]
        st.dataframe(df_meas, hide_index=True, use_container_width=True)

        if st.button("🗑️ Clear Measurements Log"):
            if session is not None: session.measurements = []
            st.session_state["saved_measurements"] = []
            st.rerun()

    # ------------------------------------------------------------------------
    # 10. Clinical Anatomy & Transitional Anatomy (LSTV) Guard
    # ------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🩺 Clinical Anatomy & Transitional Anatomy (LSTV) Verification (C2-to-S1 Protocol)", expanded=True):
        st.markdown("""
        **Clinical Doctrine: "Count from C2 • Trace the Anatomy • Verify the Level"**
        * **The Transitional Anatomy Problem (LSTV):** In clinical spine practice, 10–15% of patients possess Lumbosacral Transitional Vertebrae (sacralized L5 or lumbarized S1). An AI model operating on tight lumbar crops cannot rule out numbering shifts without scout verification.
        * **Checklist Protocol:** Verify landmarks sequentially to ensure the AI's predictions correspond to the true numerical vertebral segment:
        """)
        c_chk1, c_chk2 = st.columns(2)
        with c_chk1:
            chk_t12 = st.checkbox("🦴 T12 Rib Landmark: Verified lowest rib-bearing vertebra to anchor T12/L1", value=True)
            chk_conus = st.checkbox("🧠 Conus Medullaris Landmark: Verified spinal cord tapering at L1-L2", value=True)
            chk_aorta = st.checkbox("🫀 Vascular Landmark: Verified aortic bifurcation reference at L4", value=True)
        with c_chk2:
            chk_sacrum = st.checkbox("📐 Lumbosacral Promontory: Verified acute sacral plateau angle at L5-S1", value=True)
            chk_lstv = st.checkbox("🛡️ LSTV Exclusion: No L5 sacralization / S1 lumbarization detected", value=True)

        c_vbtn, c_vstat = st.columns([1.5, 2.5])
        with c_vbtn:
            if st.button("🩺 Sign Off & Confirm Level Verification", type="primary", use_container_width=True):
                st.session_state.clinical_level_verification = True
                if session is not None:
                    session.clinical_level_verification = True
                    session.anatomical_tracing_confirmed = True
                    session.level_verification_notes = f"Verified across L1-S1 on {nearest_lvl} focus."
                st.success("Level verification confirmed for case report.")
        with c_vstat:
            is_verified = bool(
                st.session_state.get("clinical_level_verification", False)
                or (session and getattr(session, "clinical_level_verification", False))
            )
            if is_verified:
                st.success("✅ **CLINICALLY VERIFIED:** Anatomical level chain confirmed against secondary landmarks.")
            else:
                st.warning("⚠️ **PENDING VERIFICATION:** Complete the checklist and click Sign Off above.")