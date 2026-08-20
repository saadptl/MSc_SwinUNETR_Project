from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPIDER_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider"
)


# ============================================================
# PATHS
# ============================================================

IMAGES_DIR = SPIDER_DIR / "images"
MASKS_DIR = SPIDER_DIR / "masks"

OVERVIEW_CSV = (
    SPIDER_DIR
    / "overview.csv"
)

GRADING_CSV = (
    SPIDER_DIR
    / "radiological_gradings.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHASE 3 - PART 1")
print("SPIDER DATASET VERIFICATION")
print("=" * 70)


# ============================================================
# PROJECT PATH
# ============================================================

print("\nPROJECT ROOT")
print(PROJECT_ROOT)


print("\nSPIDER DIRECTORY")
print(SPIDER_DIR)


# ============================================================
# DIRECTORY CHECK
# ============================================================

print("\nDIRECTORY STATUS")

for name, path in [
    ("SPIDER", SPIDER_DIR),
    ("IMAGES", IMAGES_DIR),
    ("MASKS", MASKS_DIR),
]:
    print(
        f"{name:10s}: "
        f"{'FOUND' if path.exists() else 'MISSING'}"
    )


# ============================================================
# CSV CHECK
# ============================================================

print("\nCSV STATUS")

for name, path in [
    ("OVERVIEW", OVERVIEW_CSV),
    ("GRADINGS", GRADING_CSV),
]:
    print(
        f"{name:10s}: "
        f"{'FOUND' if path.exists() else 'MISSING'}"
    )


# ============================================================
# OVERVIEW
# ============================================================

if OVERVIEW_CSV.exists():

    overview = pd.read_csv(
        OVERVIEW_CSV
    )

    print("\nOVERVIEW DATASET")
    print(
        "Rows   :",
        len(overview)
    )

    print(
        "Columns:",
        len(overview.columns)
    )

    print(
        "Column names:"
    )

    for column in overview.columns:
        print(
            "  -",
            column
        )


# ============================================================
# RADIOLOGICAL GRADINGS
# ============================================================

if GRADING_CSV.exists():

    gradings = pd.read_csv(
        GRADING_CSV
    )

    print("\nRADIOLOGICAL GRADINGS")

    print(
        "Rows   :",
        len(gradings)
    )

    print(
        "Columns:",
        len(gradings.columns)
    )

    print(
        "Column names:"
    )

    for column in gradings.columns:
        print(
            "  -",
            column
        )


# ============================================================
# FILE COUNTS
# ============================================================

print("\nFILE COUNTS")


if IMAGES_DIR.exists():

    image_files = [
        p
        for p in IMAGES_DIR.rglob("*")
        if p.is_file()
    ]

    print(
        "Image files:",
        len(image_files)
    )


if MASKS_DIR.exists():

    mask_files = [
        p
        for p in MASKS_DIR.rglob("*")
        if p.is_file()
    ]

    print(
        "Mask files:",
        len(mask_files)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATASET VERIFICATION COMPLETE")
print("=" * 70)