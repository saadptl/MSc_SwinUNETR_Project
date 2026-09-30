from pathlib import Path
import pydicom
import numpy as np

ROOT = Path(
    r"dataset\rsna-2024-lumbar-spine-degenerative-classification"
    r"\train_images\11943292"
)

def orientation_name(iop):
    if not iop or len(iop) != 6:
        return "Unknown"

    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row, col)

    axis = np.argmax(np.abs(normal))
    value = abs(normal[axis])

    if value < 0.8:
        return "Oblique"

    return {
        0: "Sagittal (normal ≈ X)",
        1: "Coronal (normal ≈ Y)",
        2: "Axial (normal ≈ Z)",
    }[axis]


def safe(ds, name, default="N/A"):
    value = getattr(ds, name, default)

    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]

    return str(value)


print("=" * 80)
print("RSNA DICOM STUDY / SERIES AUDIT")
print("=" * 80)
print(f"Patient folder: {ROOT}")
print()

series_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

for series_dir in series_dirs:

    files = sorted(series_dir.glob("*.dcm"))

    print("=" * 80)
    print(f"SERIES UID/FOLDER: {series_dir.name}")
    print(f"DICOM FILES: {len(files)}")

    if not files:
        print("No DICOM files found.")
        continue

    datasets = []

    for f in files:
        try:
            ds = pydicom.dcmread(
                str(f),
                stop_before_pixels=True,
                force=True
            )
            datasets.append(ds)
        except Exception as e:
            print(f"Could not read {f.name}: {e}")

    if not datasets:
        print("No readable DICOM metadata.")
        continue

    ds = datasets[0]

    print()
    print("--- IDENTIFICATION ---")
    print("PatientID:       ", safe(ds, "PatientID"))
    print("StudyInstanceUID:", safe(ds, "StudyInstanceUID"))
    print("SeriesInstanceUID:", safe(ds, "SeriesInstanceUID"))
    print("SeriesNumber:    ", safe(ds, "SeriesNumber"))
    print("SeriesDescription:", safe(ds, "SeriesDescription"))
    print("ProtocolName:    ", safe(ds, "ProtocolName"))
    print("Modality:        ", safe(ds, "Modality"))
    print("BodyPartExamined:", safe(ds, "BodyPartExamined"))
    print("ImageType:       ", safe(ds, "ImageType"))

    print()
    print("--- IMAGE GEOMETRY ---")
    print("Rows:            ", safe(ds, "Rows"))
    print("Columns:         ", safe(ds, "Columns"))
    print("PixelSpacing:    ", safe(ds, "PixelSpacing"))
    print("SliceThickness:  ", safe(ds, "SliceThickness"))
    print("SpacingBetweenSlices:",
          safe(ds, "SpacingBetweenSlices"))
    print("PatientPosition: ", safe(ds, "PatientPosition"))

    iop = getattr(ds, "ImageOrientationPatient", None)

    print()
    print("--- ORIENTATION ---")
    print("ImageOrientationPatient:", safe(ds, "ImageOrientationPatient"))
    print("Orientation:", orientation_name(iop))

    if iop and len(iop) == 6:
        row = np.asarray(iop[:3], dtype=float)
        col = np.asarray(iop[3:], dtype=float)
        normal = np.cross(row, col)

        print("Row direction:   ", np.round(row, 6))
        print("Column direction:", np.round(col, 6))
        print("Slice normal:    ", np.round(normal, 6))

    print()
    print("--- SLICE POSITIONS ---")

    positions = []

    for d in datasets:
        ipp = getattr(d, "ImagePositionPatient", None)

        if ipp and len(ipp) == 3:
            positions.append(
                np.asarray(ipp, dtype=float)
            )

    if iop and len(iop) == 6 and positions:

        row = np.asarray(iop[:3], dtype=float)
        col = np.asarray(iop[3:], dtype=float)
        normal = np.cross(row, col)

        coords = [
            float(np.dot(pos, normal))
            for pos in positions
        ]

        coords_sorted = sorted(coords)

        print(
            "Physical slice coordinates:",
            np.round(coords_sorted, 3).tolist()
        )

        if len(coords_sorted) > 1:
            diffs = np.diff(coords_sorted)
            diffs = np.abs(diffs)
            diffs = diffs[diffs > 1e-5]

            if len(diffs):
                print(
                    "Median physical slice spacing:",
                    float(np.median(diffs)),
                    "mm"
                )
                print(
                    "Min spacing:",
                    float(np.min(diffs)),
                    "mm"
                )
                print(
                    "Max spacing:",
                    float(np.max(diffs)),
                    "mm"
                )

    print()
    print("--- FILE EXAMPLES ---")

    for f in files[:3]:
        print(" ", f.name)

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)