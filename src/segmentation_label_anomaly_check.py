from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# PHASE 3 - PART 3A
# UNEXPECTED LABEL INVESTIGATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASKS_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider"
    / "masks"
)


TARGET_LABELS = {
    9,
    209,
}


mask_files = sorted(
    MASKS_DIR.rglob("*.mha")
)


print("=" * 75)
print("PHASE 3 - PART 3A")
print("UNEXPECTED LABEL INVESTIGATION")
print("=" * 75)


found = []


for mask_path in mask_files:

    image = sitk.ReadImage(
        str(mask_path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    labels = set(
        np.unique(array).tolist()
    )

    present = (
        labels
        &
        TARGET_LABELS
    )

    if present:

        counts = {}

        for label in sorted(present):

            counts[label] = int(
                np.count_nonzero(
                    array == label
                )
            )

        found.append(
            (
                mask_path.name,
                sorted(present),
                counts,
            )
        )


print(
    "\nCases containing labels 9/209:",
    len(found)
)


for filename, labels, counts in found:

    print("\n" + "-" * 60)

    print(
        "File:",
        filename
    )

    print(
        "Unexpected labels:",
        labels
    )

    for label, count in counts.items():

        print(
            f"Label {label}: "
            f"{count:,} voxels"
        )


print("\n" + "=" * 75)
print("INVESTIGATION COMPLETE")
print("=" * 75)