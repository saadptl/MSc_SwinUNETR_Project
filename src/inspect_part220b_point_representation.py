"""
PART 2.20B POINT REPRESENTATION DIAGNOSTIC

Purpose:
    Inspect the exact point records returned by the validated
    Part 2.20B physical-space geometry pipeline.

Important:
    - READ ONLY
    - No training
    - No checkpoint loading/modification
    - No XAI
    - No geometry modification
    - No fabricated points
    - Uses the existing Part 2.20B module exactly as-is

Canonical geometry module:
    segmentation_rsna_part220b_geometry_corrected_training.py
"""

from pathlib import Path
import sys
import json

import numpy as np


# =========================================================
# PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =========================================================
# IMPORTANT:
# EXISTING VALIDATED PART 2.20B MODULE
# =========================================================

import segmentation_rsna_part220b_geometry_corrected_training as part220b


# =========================================================
# CASE CONFIGURATION
# =========================================================

STUDY_ID = 7143189
SERIES_ID = 3219733239


# =========================================================
# HELPER
# =========================================================

def print_separator(title=None):

    print()
    print("=" * 80)

    if title is not None:
        print(title)

        print("=" * 80)


# =========================================================
# MAIN
# =========================================================

def main():

    print_separator(
        "PART 2.20B POINT REPRESENTATION DIAGNOSTIC"
    )

    print(
        "Purpose:"
    )

    print(
        "Inspect the exact annotation point records returned "
        "by the existing Part 2.20B physical-space pipeline."
    )

    print()

    print(
        "No training will be performed."
    )

    print(
        "No checkpoint will be loaded."
    )

    print(
        "No geometry will be modified."
    )

    print()

    print(
        "Study ID:",
        STUDY_ID
    )

    print(
        "Series ID:",
        SERIES_ID
    )

    print(
        "Geometry module:"
    )

    print(
        "segmentation_rsna_part220b_geometry_corrected_training"
    )


    # =====================================================
    # LOAD MANIFEST
    # =====================================================

    print_separator(
        "1. LOADING MANIFEST"
    )

    manifest = part220b.load_manifest()

    print(
        "Manifest type:",
        type(manifest)
    )

    print(
        "Manifest rows:",
        len(manifest)
    )

    print()

    print(
        "Manifest columns:"
    )

    for column in manifest.columns:

        print(
            "  -",
            column
        )


    # =====================================================
    # FILTER CURRENT CASE
    # =====================================================

    print_separator(
        "2. FILTERING CURRENT STUDY + SERIES"
    )

    case_manifest = manifest[
        (
            manifest["study_id"]
            .astype(str)
            == str(STUDY_ID)
        )
        &
        (
            manifest["series_id"]
            .astype(str)
            == str(SERIES_ID)
        )
    ].copy()


    print(
        "Case manifest rows:",
        len(case_manifest)
    )


    if case_manifest.empty:

        raise RuntimeError(
            "No manifest rows found for "
            f"Study {STUDY_ID}, "
            f"Series {SERIES_ID}"
        )


    print()

    print(
        "Case manifest:"
    )

    print(
        case_manifest.to_string(
            index=False
        )
    )


    # =====================================================
    # LOAD CASE THROUGH PART 2.20B
    # =====================================================

    print_separator(
        "3. CALLING PART 2.20B load_case()"
    )

    case_result = part220b.load_case(
        STUDY_ID,
        SERIES_ID,
        case_manifest,
    )


    print(
        "Returned object type:",
        type(case_result)
    )


    if not isinstance(
        case_result,
        (tuple, list)
    ):

        raise RuntimeError(
            "Unexpected return type from "
            "part220b.load_case(): "
            f"{type(case_result)}"
        )


    print(
        "Number of returned objects:",
        len(case_result)
    )


    # =====================================================
    # UNPACK CASE
    # =====================================================

    print_separator(
        "4. UNPACKING load_case() RESULT"
    )


    if len(case_result) == 3:

        image, points, geometry = (
            case_result
        )

        extra_objects = []

    else:

        image, points, geometry, *extra_objects = (
            case_result
        )


    # =====================================================
    # IMAGE
    # =====================================================

    print(
        "Image type:",
        type(image)
    )

    image_np = np.asarray(
        image
    )

    print(
        "Image shape:",
        image_np.shape
    )

    print(
        "Image dtype:",
        image_np.dtype
    )

    print(
        "Image ndim:",
        image_np.ndim
    )

    if image_np.size > 0:

        print(
            "Image minimum:",
            float(
                np.min(image_np)
            )
        )

        print(
            "Image maximum:",
            float(
                np.max(image_np)
            )
        )

        print(
            "Image mean:",
            float(
                np.mean(image_np)
            )
        )


    # =====================================================
    # POINT OBJECT
    # =====================================================

    print_separator(
        "5. POINT OBJECT"
    )

    print(
        "Points type:",
        type(points)
    )

    try:

        print(
            "Number of points:",
            len(points)
        )

    except TypeError:

        print(
            "Points object does not have len()."
        )


    # =====================================================
    # PRINT EVERY POINT
    # =====================================================

    print_separator(
        "6. RAW POINT RECORDS"
    )


    for index, point in enumerate(
        points
    ):

        print()
        print(
            "-" * 70
        )

        print(
            "POINT INDEX:",
            index
        )

        print(
            "Point type:",
            type(point)
        )

        print(
            "Raw point:"
        )

        print(
            repr(point)
        )


        if isinstance(
            point,
            dict
        ):

            print()

            print(
                "Dictionary keys:"
            )

            for key in point.keys():

                print(
                    "  -",
                    key
                )


            print()

            print(
                "Dictionary values:"
            )

            for key, value in point.items():

                print(
                    f"  {key!r}: "
                    f"{value!r} "
                    f"(type={type(value).__name__})"
                )


            # ---------------------------------------------
            # COMMON COORDINATE FIELDS
            # ---------------------------------------------

            print()

            print(
                "Coordinate fields detected:"
            )


            for key in (
                "x",
                "y",
                "z",
                "px",
                "py",
                "pz",
                "physical_x",
                "physical_y",
                "physical_z",
                "voxel_x",
                "voxel_y",
                "voxel_z",
                "model_x",
                "model_y",
                "model_z",
                "row",
                "col",
                "slice",
                "class_id",
            ):

                if key in point:

                    print(
                        f"  {key}: "
                        f"{point[key]!r}"
                    )


    # =====================================================
    # POINT CLASS DISTRIBUTION
    # =====================================================

    print_separator(
        "7. POINT CLASS DISTRIBUTION"
    )


    class_counts = {}


    for point in points:

        if not isinstance(
            point,
            dict
        ):

            continue


        class_id = point.get(
            "class_id"
        )


        if class_id is None:

            class_id = "MISSING"


        class_id_string = str(
            class_id
        )


        class_counts[
            class_id_string
        ] = (
            class_counts.get(
                class_id_string,
                0
            )
            + 1
        )


    if class_counts:

        for class_id, count in (
            class_counts.items()
        ):

            print(
                f"class_id={class_id}: "
                f"{count} point(s)"
            )

    else:

        print(
            "No class_id values found."
        )


    # =====================================================
    # COORDINATE RANGE ANALYSIS
    # =====================================================

    print_separator(
        "8. COORDINATE RANGE ANALYSIS"
    )


    coordinate_records = []


    for index, point in enumerate(
        points
    ):

        if not isinstance(
            point,
            dict
        ):

            continue


        if not all(
            key in point
            for key in (
                "x",
                "y",
                "z",
            )
        ):

            continue


        try:

            x = float(
                point["x"]
            )

            y = float(
                point["y"]
            )

            z = float(
                point["z"]
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        coordinate_records.append(
            {
                "index": index,
                "class_id": point.get(
                    "class_id"
                ),
                "x": x,
                "y": y,
                "z": z,
            }
        )


    if coordinate_records:

        xs = np.array(
            [
                record["x"]
                for record
                in coordinate_records
            ],
            dtype=np.float64,
        )

        ys = np.array(
            [
                record["y"]
                for record
                in coordinate_records
            ],
            dtype=np.float64,
        )

        zs = np.array(
            [
                record["z"]
                for record
                in coordinate_records
            ],
            dtype=np.float64,
        )


        print(
            "Number of coordinate records:",
            len(coordinate_records)
        )

        print()

        print(
            "X range:",
            float(xs.min()),
            "to",
            float(xs.max())
        )

        print(
            "Y range:",
            float(ys.min()),
            "to",
            float(ys.max())
        )

        print(
            "Z range:",
            float(zs.min()),
            "to",
            float(zs.max())
        )


        print()

        print(
            "Expected model volume:"
        )

        print(
            "Z:",
            0,
            "to",
            63
        )

        print(
            "Y:",
            0,
            "to",
            95
        )

        print(
            "X:",
            0,
            "to",
            95
        )


        print()

        print(
            "Coordinate bounds check:"
        )


        for record in coordinate_records:

            z = record["z"]
            y = record["y"]
            x = record["x"]


            z_inside = (
                0 <= z < 64
            )

            y_inside = (
                0 <= y < 96
            )

            x_inside = (
                0 <= x < 96
            )


            print(
                f"Point {record['index']} "
                f"class={record['class_id']} "
                f""
                f"(z={z:.4f}, "
                f"y={y:.4f}, "
                f"x={x:.4f}) "
                f""
                f"inside_model_grid="
                f"{z_inside and y_inside and x_inside}"
            )


    else:

        print(
            "No x/y/z coordinate records "
            "were detected."
        )


    # =====================================================
    # GEOMETRY OBJECT
    # =====================================================

    print_separator(
        "9. GEOMETRY OBJECT"
    )

    print(
        "Geometry type:",
        type(geometry)
    )

    print()

    print(
        "Geometry representation:"
    )

    print(
        repr(geometry)
    )


    # =====================================================
    # GEOMETRY DICTIONARY
    # =====================================================

    if isinstance(
        geometry,
        dict
    ):

        print()

        print(
            "Geometry dictionary keys:"
        )

        for key in geometry.keys():

            print(
                "  -",
                key
            )


        print()

        print(
            "Geometry dictionary values:"
        )

        for key, value in geometry.items():

            print(
                f"  {key!r}: "
                f"{value!r} "
                f"(type={type(value).__name__})"
            )


    # =====================================================
    # EXTRA RETURNED OBJECTS
    # =====================================================

    if extra_objects:

        print_separator(
            "10. ADDITIONAL load_case() OBJECTS"
        )


        print(
            "Number of additional objects:",
            len(extra_objects)
        )


        for index, obj in enumerate(
            extra_objects
        ):

            print()

            print(
                "Additional object",
                index
            )

            print(
                "Type:",
                type(obj)
            )

            print(
                "Representation:"
            )

            print(
                repr(obj)
            )


    # =====================================================
    # FINAL INTERPRETATION HELP
    # =====================================================

    print_separator(
        "11. DIAGNOSTIC SUMMARY"
    )


    print(
        "This script has NOT modified the geometry pipeline."
    )

    print(
        "This script has NOT modified the checkpoint."
    )

    print(
        "This script has NOT generated an XAI map."
    )

    print()


    print(
        "Important:"
    )

    print(
        "We will use the raw point records and geometry "
        "information above to determine whether x/y/z are "
        "already model-grid coordinates or represent another "
        "coordinate system."
    )

    print()

    print(
        "Do NOT manually convert the points yet."
    )

    print(
        "Do NOT change Part 2.20B yet."
    )

    print(
        "Do NOT change Part 4.6 yet."
    )


    # =====================================================
    # OPTIONAL MACHINE-READABLE DIAGNOSTIC
    # =====================================================

    diagnostic = {

        "study_id": STUDY_ID,

        "series_id": SERIES_ID,

        "image_shape": list(
            image_np.shape
        ),

        "number_of_points": len(
            points
        ),

        "class_counts": class_counts,

        "coordinate_records": (
            coordinate_records
        ),
    }


    diagnostic_path = (
        ROOT
        / "outputs"
        / "segmentation"
        / "part220b_point_diagnostic.json"
    )


    diagnostic_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        diagnostic_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            diagnostic,
            f,
            indent=2,
        )


    print()

    print(
        "Diagnostic JSON saved to:"
    )

    print(
        diagnostic_path
    )


    print()

    print_separator(
        "DIAGNOSTIC COMPLETE"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()