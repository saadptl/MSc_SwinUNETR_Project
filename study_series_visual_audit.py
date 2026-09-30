from pathlib import Path
import math
import cv2
import numpy as np
import pydicom

ROOT = Path(
    r"dataset\rsna-2024-lumbar-spine-degenerative-classification"
    r"\train_images\11943292"
)

OUT = Path("outputs") / "dashboard" / "series_audit"
OUT.mkdir(parents=True, exist_ok=True)


def normalize(img):
    img = img.astype(np.float32)

    lo, hi = np.percentile(img, [1, 99])

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip((img - lo) / (hi - lo), 0, 1)

    return (img * 255).astype(np.uint8)


def read_series(series_dir):

    datasets = []

    for f in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), force=True)

            if hasattr(ds, "PixelData"):
                datasets.append(ds)

        except Exception:
            pass

    if not datasets:
        return []

    # Sort using ImagePositionPatient when available
    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)

        if ipp is not None:
            return tuple(float(x) for x in ipp)

        return int(getattr(ds, "InstanceNumber", 0) or 0)

    datasets.sort(key=sort_key)

    images = []

    for ds in datasets:

        try:
            img = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))

            img = img * slope + intercept

            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                img = img.max() - img

            images.append(img)

        except Exception:
            pass

    return images


def make_contact_sheet(images, title, output):

    if not images:
        return

    thumb_w = 220
    thumb_h = 220

    columns = 5
    rows = math.ceil(len(images) / columns)

    header_h = 70

    canvas = np.zeros(
        (
            rows * thumb_h + header_h,
            columns * thumb_w,
            3
        ),
        dtype=np.uint8
    )

    cv2.putText(
        canvas,
        title,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    for i, img in enumerate(images):

        img8 = normalize(img)

        h, w = img8.shape

        scale = min(
            (thumb_w - 10) / w,
            (thumb_h - 35) / h
        )

        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))

        thumb = cv2.resize(
            img8,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

        x = (i % columns) * thumb_w
        y = header_h + (i // columns) * thumb_h

        x0 = x + (thumb_w - nw) // 2
        y0 = y + 5

        canvas[y0:y0+nh, x0:x0+nw] = np.stack(
            [thumb, thumb, thumb],
            axis=-1
        )

        cv2.putText(
            canvas,
            f"{i + 1}/{len(images)}",
            (x + 10, y + thumb_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(str(output), canvas)


for series in sorted(ROOT.iterdir()):

    if not series.is_dir():
        continue

    images = read_series(series)

    print(
        f"{series.name}: "
        f"{len(images)} readable slices"
    )

    if images:

        output = OUT / f"{series.name}_contact_sheet.png"

        make_contact_sheet(
            images,
            f"Series {series.name} — {len(images)} slices",
            output
        )

        print("  Saved:", output)


print()
print("AUDIT COMPLETE")