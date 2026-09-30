from pathlib import Path
import numpy as np
import pydicom

ROOT = Path(
    r"dataset\rsna-2024-lumbar-spine-degenerative-classification"
    r"\train_images\11943292"
)

SERIES = ["3800798510", "403244853"]


def get_series_info(name):
    folder = ROOT / name
    files = sorted(folder.glob("*.dcm"))

    dsets = []

    for f in files:
        ds = pydicom.dcmread(
            str(f),
            stop_before_pixels=True,
            force=True
        )

        if hasattr(ds, "ImagePositionPatient"):
            dsets.append(ds)

    ds0 = dsets[0]

    row = np.asarray(
        ds0.ImageOrientationPatient[:3],
        dtype=float
    )

    col = np.asarray(
        ds0.ImageOrientationPatient[3:],
        dtype=float
    )

    normal = np.cross(row, col)
    normal = normal / np.linalg.norm(normal)

    positions = np.array(
        [
            np.asarray(
                ds.ImagePositionPatient,
                dtype=float
            )
            for ds in dsets
        ]
    )

    # Physical coordinate along this series' own slice normal
    coords = positions @ normal

    order = np.argsort(coords)

    positions = positions[order]
    coords = coords[order]

    spacing = np.diff(coords)

    pixel_spacing = np.asarray(
        ds0.PixelSpacing,
        dtype=float
    )

    rows = int(ds0.Rows)
    cols = int(ds0.Columns)

    # Four physical corners of each image plane
    corners = []

    for pos in positions:
        p = np.asarray(pos, dtype=float)

        corners.extend([
            p,
            p + row * pixel_spacing[1] * (cols - 1),
            p + col * pixel_spacing[0] * (rows - 1),
            p
            + row * pixel_spacing[1] * (cols - 1)
            + col * pixel_spacing[0] * (rows - 1),
        ])

    corners = np.asarray(corners)

    # Approximate physical extent including slice direction
    first = positions[0]
    last = positions[-1]

    all_points = np.vstack([
        corners,
        corners + normal * 4.4,
        corners - normal * 4.4,
    ])

    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)

    center = positions.mean(axis=0)

    print("=" * 80)
    print(f"SERIES: {name}")
    print("=" * 80)

    print("Description: ", getattr(ds0, "SeriesDescription", "N/A"))
    print("Rows/Cols:   ", rows, cols)
    print("Slices:      ", len(dsets))
    print("PixelSpacing:", list(pixel_spacing))
    print()

    print("Row direction:")
    print(np.round(row, 6))

    print("Column direction:")
    print(np.round(col, 6))

    print("Slice normal:")
    print(np.round(normal, 6))

    print()

    print("First ImagePositionPatient:")
    print(np.round(first, 3))

    print("Last ImagePositionPatient:")
    print(np.round(last, 3))

    print("Series center:")
    print(np.round(center, 3))

    print()

    print("Patient-coordinate bounding box:")
    print("X:", round(float(bbox_min[0]), 3),
          "to", round(float(bbox_max[0]), 3))
    print("Y:", round(float(bbox_min[1]), 3),
          "to", round(float(bbox_max[1]), 3))
    print("Z:", round(float(bbox_min[2]), 3),
          "to", round(float(bbox_max[2]), 3))

    if len(spacing):
        print()
        print(
            "Median slice spacing:",
            round(float(np.median(np.abs(spacing))), 5),
            "mm"
        )

    print()


infos = {}

for name in SERIES:
    get_series_info(name)

print("=" * 80)
print("ORIENTATION COMPARISON")
print("=" * 80)

dsets = {}

for name in SERIES:
    files = sorted((ROOT / name).glob("*.dcm"))

    ds = pydicom.dcmread(
        str(files[0]),
        stop_before_pixels=True,
        force=True
    )

    row = np.asarray(
        ds.ImageOrientationPatient[:3],
        dtype=float
    )

    col = np.asarray(
        ds.ImageOrientationPatient[3:],
        dtype=float
    )

    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)

    dsets[name] = {
        "row": row,
        "col": col,
        "normal": normal,
    }

n1 = dsets[SERIES[0]]["normal"]
n2 = dsets[SERIES[1]]["normal"]

dot = float(np.clip(abs(np.dot(n1, n2)), 0, 1))
angle = np.degrees(np.arccos(dot))

print("Angle between slice normals:",
      round(float(angle), 4), "degrees")

print()
print("If the angle is small, the acquisitions are"
      " similarly oriented.")
print("If the angle is substantial, keep the series"
      " separate until registration is established.")

print()
print("AUDIT COMPLETE")