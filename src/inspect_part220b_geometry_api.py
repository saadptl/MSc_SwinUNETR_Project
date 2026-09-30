import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import segmentation_rsna_part220b_geometry_corrected_training as part220b


print("=" * 80)
print("PART 2.20B GEOMETRY API INSPECTION")
print("=" * 80)

print("\nModule:")
print(part220b.__file__)

print("\nFunctions containing geometry/canonical/patient:")
for name in sorted(dir(part220b)):

    lower = name.lower()

    if (
        "geometry" in lower
        or "canonical" in lower
        or "patient" in lower
        or "physical" in lower
        or "native" in lower
    ):

        obj = getattr(
            part220b,
            name,
        )

        if callable(obj):

            print(
                f"  FUNCTION : {name}"
            )

        else:

            print(
                f"  OBJECT   : {name} "
                f"({type(obj).__name__})"
            )


print("\n" + "=" * 80)
print("LOADING MANIFEST")
print("=" * 80)

manifest = part220b.load_manifest()

print(
    f"Manifest rows: {len(manifest)}"
)

selected = part220b.select_validation_series(
    manifest
)

print(
    f"Selected validation series: "
    f"{len(selected)}"
)

validation_keys = {
    (
        str(row["study_id"]),
        str(row["series_id"]),
    )
    for _, row in selected.iterrows()
}

validation_manifest = manifest[
    manifest.apply(
        lambda row: (
            str(row["study_id"]),
            str(row["series_id"]),
        ) in validation_keys,
        axis=1,
    )
].copy()

cases = part220b.build_case_index(
    validation_manifest
)

case_order = {
    (
        str(row["study_id"]),
        str(row["series_id"]),
    ): index
    for index, row in selected.iterrows()
}

cases = sorted(
    cases,
    key=lambda case:
    case_order.get(
        (
            str(case["study_id"]),
            str(case["series_id"]),
        ),
        10000,
    ),
)

print(
    f"Validation cases: {len(cases)}"
)


print("\n" + "=" * 80)
print("INSPECTING FIRST VALIDATION CASE")
print("=" * 80)

case = cases[0]

study_id = str(
    case["study_id"]
)

series_id = str(
    case["series_id"]
)

print(
    f"Study : {study_id}"
)

print(
    f"Series: {series_id}"
)

case_rows = case["points"]

if not isinstance(
    case_rows,
    pd.DataFrame,
):

    case_rows = pd.DataFrame(
        case_rows
    )

case_rows = case_rows.reset_index(
    drop=True
)

image, transformed_points, geometry = (
    part220b.load_case(
        study_id,
        series_id,
        case_rows,
    )
)

print("\nImage shape:")
print(
    getattr(
        image,
        "shape",
        None,
    )
)

print("\nGeometry type:")
print(
    type(geometry)
)

print("\nGeometry repr:")
print(
    repr(geometry)
)

print("\nGeometry attributes:")
for name in sorted(
    dir(geometry)
):

    if name.startswith("_"):
        continue

    try:

        value = getattr(
            geometry,
            name,
        )

        if callable(value):

            print(
                f"  FUNCTION : {name}"
            )

        else:

            text = repr(value)

            if len(text) > 500:

                text = (
                    text[:500]
                    + " ..."
                )

            print(
                f"  ATTRIBUTE: {name}"
            )
            print(
                f"             {text}"
            )

    except Exception as exc:

        print(
            f"  {name}: "
            f"<ERROR {exc}>"
        )


print("\n" + "=" * 80)
print("FIRST TRANSFORMED POINT")
print("=" * 80)

if transformed_points:

    print(
        transformed_points[0]
    )

else:

    print(
        "No transformed points."
    )


print("\n" + "=" * 80)
print("PART 2.20B SOURCE FILE LOCATION")
print("=" * 80)

print(
    part220b.__file__
)

print("\nInspection complete.")