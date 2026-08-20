from pathlib import Path
import random
import shutil

import pandas as pd


# ============================================================
# PHASE 3 - PART 9
# PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("=" * 75)
print("PHASE 3 - PART 9")
print("PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 75)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

MRI_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "images_preprocessed"
)

MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "masks_preprocessed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "spider_processed"
    / "segmentation_split"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "split"
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\nPROJECT ROOT")
print(PROJECT_ROOT)

print("\nMRI DIRECTORY")
print(MRI_DIR)

print("\nMASK DIRECTORY")
print(MASK_DIR)

print("\nSPLIT OUTPUT")
print(OUTPUT_ROOT)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


print("\n" + "=" * 75)
print("SPLIT CONFIGURATION")
print("=" * 75)

print(
    "Train ratio:",
    TRAIN_RATIO
)

print(
    "Validation ratio:",
    VAL_RATIO
)

print(
    "Test ratio:",
    TEST_RATIO
)

print(
    "Random seed:",
    RANDOM_SEED
)


# ============================================================
# VALIDATE RATIOS
# ============================================================

ratio_sum = (
    TRAIN_RATIO
    +
    VAL_RATIO
    +
    TEST_RATIO
)

if abs(ratio_sum - 1.0) > 1e-8:

    raise RuntimeError(
        "Train/validation/test ratios "
        "must sum to 1.0."
    )


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not MRI_DIR.exists():

    raise FileNotFoundError(
        f"MRI directory not found:\n{MRI_DIR}"
    )

if not MASK_DIR.exists():

    raise FileNotFoundError(
        f"Mask directory not found:\n{MASK_DIR}"
    )


# ============================================================
# FIND FILES
# ============================================================

mri_files = sorted(
    MRI_DIR.glob("*.mha")
)

mask_files = sorted(
    MASK_DIR.glob("*.mha")
)


print("\n" + "=" * 75)
print("DATASET FILE COUNTS")
print("=" * 75)

print(
    "MRI files:",
    len(mri_files)
)

print(
    "Mask files:",
    len(mask_files)
)


if len(mri_files) == 0:

    raise RuntimeError(
        "No preprocessed MRI files found."
    )

if len(mask_files) == 0:

    raise RuntimeError(
        "No preprocessed mask files found."
    )


# ============================================================
# BUILD MASK LOOKUP
# ============================================================

mask_lookup = {
    path.name: path
    for path in mask_files
}


# ============================================================
# CREATE MRI / MASK PAIRS
# ============================================================

records = []

missing_masks = []


for mri_path in mri_files:

    mask_path = mask_lookup.get(
        mri_path.name
    )

    if mask_path is None:

        missing_masks.append(
            mri_path.name
        )

        continue

    records.append(
        {
            "filename": mri_path.name,
            "mri_path": str(mri_path),
            "mask_path": str(mask_path),
        }
    )


print("\n" + "=" * 75)
print("MRI / MASK PAIRING")
print("=" * 75)

print(
    "Matching pairs:",
    len(records)
)

print(
    "Missing masks:",
    len(missing_masks)
)


if missing_masks:

    print("\nMissing mask files:")

    for filename in missing_masks[:20]:

        print(
            " ",
            filename
        )

    raise RuntimeError(
        "MRI/mask pairing is incomplete."
    )


# ============================================================
# EXTRACT PATIENT ID
# ============================================================

def extract_patient_id(
    filename: str
) -> str:

    stem = Path(
        filename
    ).stem

    # Expected examples:
    #
    # 100_t1
    # 100_t2
    # 152_t2_SPACE
    #
    # Patient ID is the first
    # underscore-separated component.

    patient_id = (
        stem.split("_")[0]
    )

    if not patient_id:

        raise ValueError(
            f"Unable to extract patient ID "
            f"from {filename}"
        )

    return patient_id


for record in records:

    record["patient_id"] = (
        extract_patient_id(
            record["filename"]
        )
    )


# ============================================================
# DATASET SUMMARY BY PATIENT
# ============================================================

df = pd.DataFrame(
    records
)


patients = sorted(
    df["patient_id"]
    .unique()
)


print("\n" + "=" * 75)
print("PATIENT INFORMATION")
print("=" * 75)

print(
    "Unique patients:",
    len(patients)
)

print(
    "Total MRI volumes:",
    len(df)
)


patient_volume_counts = (
    df.groupby(
        "patient_id"
    )
    .size()
)


print(
    "Minimum volumes/patient:",
    int(
        patient_volume_counts.min()
    )
)

print(
    "Maximum volumes/patient:",
    int(
        patient_volume_counts.max()
    )
)

print(
    "Average volumes/patient:",
    round(
        patient_volume_counts.mean(),
        2
    )
)


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

random.seed(
    RANDOM_SEED
)

shuffled_patients = patients.copy()

random.shuffle(
    shuffled_patients
)


num_patients = len(
    shuffled_patients
)


train_count = int(
    num_patients
    * TRAIN_RATIO
)

val_count = int(
    num_patients
    * VAL_RATIO
)

# Remaining patients go to test.
test_count = (
    num_patients
    -
    train_count
    -
    val_count
)


train_patients = (
    shuffled_patients[
        :train_count
    ]
)

val_patients = (
    shuffled_patients[
        train_count:
        train_count + val_count
    ]
)

test_patients = (
    shuffled_patients[
        train_count + val_count:
    ]
)


print("\n" + "=" * 75)
print("PATIENT SPLIT")
print("=" * 75)

print(
    "Train patients:",
    len(train_patients)
)

print(
    "Validation patients:",
    len(val_patients)
)

print(
    "Test patients:",
    len(test_patients)
)

print(
    "Total:",
    (
        len(train_patients)
        +
        len(val_patients)
        +
        len(test_patients)
    )
)


# ============================================================
# VERIFY PATIENT DISJOINTNESS
# ============================================================

train_set = set(
    train_patients
)

val_set = set(
    val_patients
)

test_set = set(
    test_patients
)


train_val_overlap = (
    train_set
    &
    val_set
)

train_test_overlap = (
    train_set
    &
    test_set
)

val_test_overlap = (
    val_set
    &
    test_set
)


print("\n" + "=" * 75)
print("PATIENT LEAKAGE CHECK")
print("=" * 75)

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    train_val_overlap
    or train_test_overlap
    or val_test_overlap
):

    raise RuntimeError(
        "Patient leakage detected!"
    )


print(
    "✓ No patient appears in more than one split."
)


# ============================================================
# ASSIGN SPLIT TO EVERY VOLUME
# ============================================================

def assign_split(
    patient_id: str
) -> str:

    if patient_id in train_set:

        return "train"

    if patient_id in val_set:

        return "validation"

    if patient_id in test_set:

        return "test"

    raise RuntimeError(
        f"Patient {patient_id} "
        "does not belong to a split."
    )


df["split"] = (
    df["patient_id"]
    .apply(assign_split)
)


# ============================================================
# VOLUME STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("VOLUME SPLIT")
print("=" * 75)


volume_counts = (
    df["split"]
    .value_counts()
)


for split_name in [
    "train",
    "validation",
    "test",
]:

    count = int(
        volume_counts.get(
            split_name,
            0
        )
    )

    percentage = (
        count
        /
        len(df)
        *
        100
    )

    print(
        f"{split_name.capitalize():12s}: "
        f"{count:3d} volumes "
        f"({percentage:.2f}%)"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split_name in [
    "train",
    "validation",
    "test",
]:

    (
        OUTPUT_ROOT
        / split_name
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_ROOT
        / split_name
        / "masks"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY FILES
# ============================================================

print("\n" + "=" * 75)
print("CREATING SPLIT DATASET")
print("=" * 75)

print(
    "This step copies the preprocessed "
    "MRI and mask files."
)

print(
    "Original preprocessed files "
    "will NOT be modified."
)


for index, row in df.iterrows():

    split_name = row["split"]

    source_mri = Path(
        row["mri_path"]
    )

    source_mask = Path(
        row["mask_path"]
    )

    destination_mri = (
        OUTPUT_ROOT
        / split_name
        / "images"
        / source_mri.name
    )

    destination_mask = (
        OUTPUT_ROOT
        / split_name
        / "masks"
        / source_mask.name
    )

    if not destination_mri.exists():

        shutil.copy2(
            source_mri,
            destination_mri
        )

    if not destination_mask.exists():

        shutil.copy2(
            source_mask,
            destination_mask
        )

    if (
        index == 0
        or
        (index + 1) % 50 == 0
        or
        index + 1 == len(df)
    ):

        print(
            f"Copied {index + 1:3d} / "
            f"{len(df)}"
        )


# ============================================================
# VERIFY SPLIT FILES
# ============================================================

print("\n" + "=" * 75)
print("SPLIT FILE VALIDATION")
print("=" * 75)


validation_rows = []


for split_name in [
    "train",
    "validation",
    "test",
]:

    image_dir = (
        OUTPUT_ROOT
        / split_name
        / "images"
    )

    mask_dir = (
        OUTPUT_ROOT
        / split_name
        / "masks"
    )

    split_images = sorted(
        image_dir.glob("*.mha")
    )

    split_masks = sorted(
        mask_dir.glob("*.mha")
    )

    image_names = {
        path.name
        for path in split_images
    }

    mask_names = {
        path.name
        for path in split_masks
    }

    missing_split_masks = (
        image_names
        -
        mask_names
    )

    orphan_masks = (
        mask_names
        -
        image_names
    )

    split_patients = set(
        df.loc[
            df["split"] == split_name,
            "patient_id"
        ]
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "Images:",
        len(split_images)
    )

    print(
        "Masks:",
        len(split_masks)
    )

    print(
        "Patients:",
        len(split_patients)
    )

    print(
        "Missing masks:",
        len(missing_split_masks)
    )

    print(
        "Orphan masks:",
        len(orphan_masks)
    )

    if missing_split_masks:

        raise RuntimeError(
            f"{split_name}: "
            "missing masks detected."
        )

    if orphan_masks:

        raise RuntimeError(
            f"{split_name}: "
            "orphan masks detected."
        )

    validation_rows.append(
        {
            "split": split_name,
            "patients": len(
                split_patients
            ),
            "images": len(
                split_images
            ),
            "masks": len(
                split_masks
            ),
            "missing_masks": len(
                missing_split_masks
            ),
            "orphan_masks": len(
                orphan_masks
            ),
        }
    )


# ============================================================
# SECOND LEAKAGE CHECK FROM OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT DIRECTORY LEAKAGE CHECK")
print("=" * 75)


output_patient_sets = {}


for split_name in [
    "train",
    "validation",
    "test",
]:

    files = sorted(
        (
            OUTPUT_ROOT
            / split_name
            / "images"
        ).glob("*.mha")
    )

    output_patient_sets[
        split_name
    ] = {
        extract_patient_id(
            path.name
        )
        for path in files
    }


for first_split, second_split in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]:

    overlap = (
        output_patient_sets[first_split]
        &
        output_patient_sets[second_split]
    )

    print(
        f"{first_split} ∩ "
        f"{second_split}:",
        len(overlap)
    )

    if overlap:

        raise RuntimeError(
            "Patient leakage detected "
            "in split output."
        )


print(
    "✓ Output split is patient-disjoint."
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_file = (
    MANIFEST_DIR
    / "segmentation_patient_split_manifest.csv"
)

df_sorted = (
    df.sort_values(
        [
            "split",
            "patient_id",
            "filename",
        ]
    )
    .reset_index(
        drop=True
    )
)


df_sorted.to_csv(
    manifest_file,
    index=False
)


# ============================================================
# SAVE PATIENT LISTS
# ============================================================

train_patient_file = (
    MANIFEST_DIR
    / "train_patients.txt"
)

val_patient_file = (
    MANIFEST_DIR
    / "validation_patients.txt"
)

test_patient_file = (
    MANIFEST_DIR
    / "test_patients.txt"
)


train_patient_file.write_text(
    "\n".join(
        train_patients
    ),
    encoding="utf-8"
)

val_patient_file.write_text(
    "\n".join(
        val_patients
    ),
    encoding="utf-8"
)

test_patient_file.write_text(
    "\n".join(
        test_patients
    ),
    encoding="utf-8"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    MANIFEST_DIR
    / "split_summary.csv"
)

summary_df = pd.DataFrame(
    validation_rows
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SAVE HUMAN-READABLE REPORT
# ============================================================

report_file = (
    MANIFEST_DIR
    / "phase3_part9_split_report.txt"
)


train_volumes = int(
    volume_counts.get(
        "train",
        0
    )
)

val_volumes = int(
    volume_counts.get(
        "validation",
        0
    )
)

test_volumes = int(
    volume_counts.get(
        "test",
        0
    )
)


report = f"""
PHASE 3 - PART 9
PATIENT-LEVEL DATASET SPLIT
===========================

Dataset
-------
Total MRI volumes: {len(df)}
Total patients: {len(patients)}

Split ratios
------------
Train: {TRAIN_RATIO}
Validation: {VAL_RATIO}
Test: {TEST_RATIO}

Patient counts
--------------
Train patients: {len(train_patients)}
Validation patients: {len(val_patients)}
Test patients: {len(test_patients)}

Volume counts
-------------
Train volumes: {train_volumes}
Validation volumes: {val_volumes}
Test volumes: {test_volumes}

Patient leakage
---------------
Train / Validation overlap:
{len(train_val_overlap)}

Train / Test overlap:
{len(train_test_overlap)}

Validation / Test overlap:
{len(val_test_overlap)}

Result:
No patient leakage detected.

Data integrity
--------------
MRI/mask pairs before split:
{len(records)}

Missing masks:
{len(missing_masks)}

Split output contains matched
MRI/mask pairs for all partitions.

Random seed:
{RANDOM_SEED}

Important
---------
The split was performed at patient level.

No model training was performed.
No model checkpoint was created.
No existing checkpoint was modified.
"""


report_file.write_text(
    report,
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    "Dataset split:",
    OUTPUT_ROOT
)

print(
    "Manifest:",
    manifest_file
)

print(
    "Summary:",
    summary_file
)

print(
    "Training patients:",
    train_patient_file
)

print(
    "Validation patients:",
    val_patient_file
)

print(
    "Test patients:",
    test_patient_file
)

print(
    "Report:",
    report_file
)


print("\n" + "=" * 75)
print("PHASE 3 - PART 9 COMPLETE")
print("=" * 75)