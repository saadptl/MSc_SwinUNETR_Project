from pathlib import Path

TARGET = Path(__file__).resolve().parent / "segmentation_rsna_part65_point_to_pseudomask_alignment_forensics.py"

if not TARGET.exists():
    raise FileNotFoundError(f"Part 65 file not found: {TARGET}")

text = TARGET.read_text(encoding="utf-8")

HELPER = r"""
def centered_crop_local(image, mask, crop_shape):
    # Local foreground-centered crop; Part 11 does not expose centered_crop().
    import numpy as np

    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image.shape} vs {mask.shape}"
        )

    dz, dy, dx = map(int, crop_shape)
    sz, sy, sx = mask.shape

    if dz > sz or dy > sy or dx > sx:
        raise ValueError(
            f"Crop {crop_shape} is larger than volume {mask.shape}"
        )

    fg = np.argwhere(mask > 0)

    if len(fg):
        cz, cy, cx = np.mean(fg, axis=0)
    else:
        cz = (sz - 1) / 2.0
        cy = (sy - 1) / 2.0
        cx = (sx - 1) / 2.0

    def start(center, crop, size):
        s = int(round(float(center) - float(crop) / 2.0))
        return max(0, min(s, size - crop))

    z0 = start(cz, dz, sz)
    y0 = start(cy, dy, sy)
    x0 = start(cx, dx, sx)

    return (
        image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx],
        mask[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx],
        (z0, y0, x0),
    )
"""

if "def centered_crop_local(" not in text:
    marker = "def main("
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError("Could not locate main() in Part 65.")
    text = text[:pos] + HELPER + "\n\n" + text[pos:]

text = text.replace(
    "p11.centered_crop(p11.resize_3d(im,FULL,is_mask=False),fullmask)",
    "centered_crop_local(p11.resize_3d(im,FULL,is_mask=False),fullmask,CROP)"
)
text = text.replace("p11.centered_crop(", "centered_crop_local(")

if "p11.centered_crop(" in text:
    raise RuntimeError("A p11.centered_crop() call remains.")

compile(text, str(TARGET), "exec")
TARGET.write_text(text, encoding="utf-8")

print("=" * 78)
print("PART 65 API PATCH")
print("=" * 78)
print(f"Patched file : {TARGET}")
print("Added local centered_crop_local().")
print("Removed dependency on p11.centered_crop().")
print("Syntax validation : PASSED")
