from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PART 89
# RSNA CLASSIFICATION DATASET & DATALOADER PIPELINE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RSNA_ROOT = (
    ROOT
    / "dataset"
    / "rsna-2024-lumbar-spine-degenerative-classification"
)

TRAIN_IMAGES = RSNA_ROOT / "train_images"

TRAIN_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_train_manifest.csv"
)

VAL_MANIFEST = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part87_study_level_split"
    / "part87_validation_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "classification"
    / "rsna_part89_dataset_pipeline"
)

REPORT_DIR = ROOT / "reports"

BATCH_RESULTS_CSV = (
    OUTPUT_DIR
    / "part89_batch_validation_results.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "part89_dataset_pipeline_summary.json"
)

REPORT_TXT = (
    REPORT_DIR
    / "part89_dataset_pipeline_report.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 2

NUM_WORKERS = 0

# We intentionally keep this small because the RTX 2050
# has 4 GB VRAM. This is a CPU-side DataLoader test.
TARGET_DEPTH = 32
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

PREFETCH_BATCHES = 3


CONDITIONS = [
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_subarticular_stenosis",
]

LEVELS = [
    "l1_l2",
    "l2_l3",
    "l3_l4",
    "l4_l5",
    "l5_s1",
]


TARGET_COLUMNS = [
    f"{condition}_{level}"
    for condition in CONDITIONS
    for level in LEVELS
]


SEVERITY_TO_INDEX = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2,
}


INDEX_TO_SEVERITY = {
    0: "Normal/Mild",
    1: "Moderate",
    2: "Severe",
}


# ============================================================
# UTILITIES
# ============================================================

def banner(text):

    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    banner("PART 89 PATH VALIDATION")

    paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "training manifest": TRAIN_MANIFEST,
        "validation manifest": VAL_MANIFEST,
    }

    missing = []

    for name, path in paths.items():

        exists = path.exists()

        print(
            f"{name:<35}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(
            "Missing required input files/directories:\n"
            + "\n".join(missing)
        )


# ============================================================
# MANIFEST LOADING
# ============================================================

def load_manifests():

    banner("LOADING PART 87 MANIFESTS")

    train = pd.read_csv(
        TRAIN_MANIFEST
    )

    val = pd.read_csv(
        VAL_MANIFEST
    )

    print(
        f"Training studies   : {len(train)}"
    )

    print(
        f"Validation studies : {len(val)}"
    )

    required_columns = {
        "study_id",
        "series_id",
        "series_description",
        "series_path",
        "selected",
        "series_exists",
        "dicom_file_count",
    }

    required_columns.update(
        TARGET_COLUMNS
    )

    missing_train = (
        required_columns
        - set(train.columns)
    )

    missing_val = (
        required_columns
        - set(val.columns)
    )

    if missing_train:

        raise RuntimeError(
            "Training manifest is missing:\n"
            + "\n".join(
                sorted(missing_train)
            )
        )

    if missing_val:

        raise RuntimeError(
            "Validation manifest is missing:\n"
            + "\n".join(
                sorted(missing_val)
            )
        )

    return train, val


# ============================================================
# DICOM FILE DISCOVERY
# ============================================================

def discover_dicom_files(
    series_path
):

    path = Path(
        str(series_path)
    )

    if not path.exists():

        return []

    try:

        files = [
            p
            for p in path.iterdir()
            if p.is_file()
        ]

    except Exception:

        return []

    files.sort(
        key=lambda p: p.name
    )

    return files


# ============================================================
# DICOM SERIES LOADER
# ============================================================

def load_dicom_series(
    series_path
):

    files = discover_dicom_files(
        series_path
    )

    if not files:

        raise FileNotFoundError(
            f"No files found: {series_path}"
        )

    slices = []

    for path in files:

        try:

            ds = pydicom.dcmread(
                str(path),
                force=True
            )

            if not hasattr(
                ds,
                "PixelData"
            ):

                continue

            image = ds.pixel_array

            if image.ndim != 2:

                continue

            instance_number = getattr(
                ds,
                "InstanceNumber",
                None
            )

            position = getattr(
                ds,
                "ImagePositionPatient",
                None
            )

            z_position = None

            if (
                position is not None
                and len(position) >= 3
            ):

                try:

                    z_position = float(
                        position[2]
                    )

                except Exception:

                    z_position = None

            slices.append(
                {
                    "array":
                        np.asarray(
                            image,
                            dtype=np.float32
                        ),

                    "instance_number":
                        instance_number,

                    "z_position":
                        z_position,

                    "path":
                        path,
                }
            )

        except Exception:

            continue

    if not slices:

        raise RuntimeError(
            f"No readable DICOM slices: "
            f"{series_path}"
        )

    # --------------------------------------------------------
    # Slice ordering
    # --------------------------------------------------------

    if all(
        s["z_position"] is not None
        for s in slices
    ):

        slices.sort(
            key=lambda x:
                x["z_position"]
        )

    elif all(
        s["instance_number"] is not None
        for s in slices
    ):

        slices.sort(
            key=lambda x:
                int(x["instance_number"])
        )

    else:

        slices.sort(
            key=lambda x:
                x["path"].name
        )

    # --------------------------------------------------------
    # Harmonize in-plane dimensions
    # --------------------------------------------------------

    max_h = max(
        s["array"].shape[0]
        for s in slices
    )

    max_w = max(
        s["array"].shape[1]
        for s in slices
    )

    volume = np.zeros(
        (
            len(slices),
            max_h,
            max_w,
        ),
        dtype=np.float32
    )

    for i, item in enumerate(slices):

        image = item["array"]

        h, w = image.shape

        volume[
            i,
            :h,
            :w
        ] = image

    return volume


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_volume(
    volume
):

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        raise ValueError(
            "MRI volume contains no finite values."
        )

    valid = volume[
        finite
    ]

    low = np.percentile(
        valid,
        1.0
    )

    high = np.percentile(
        valid,
        99.0
    )

    if high <= low:

        low = float(
            valid.min()
        )

        high = float(
            valid.max()
        )

    clipped = np.clip(
        volume,
        low,
        high
    )

    if high > low:

        normalized = (
            clipped - low
        ) / (
            high - low
        )

    else:

        normalized = np.zeros_like(
            clipped
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# CENTER CROP / PAD
# ============================================================

def center_crop_or_pad_3d(
    volume,
    target_depth,
    target_height,
    target_width
):

    source = np.asarray(
        volume,
        dtype=np.float32
    )

    src_d, src_h, src_w = (
        source.shape
    )

    output = np.zeros(
        (
            target_depth,
            target_height,
            target_width,
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Source crop
    # --------------------------------------------------------

    d_start = max(
        0,
        (src_d - target_depth) // 2
    )

    h_start = max(
        0,
        (src_h - target_height) // 2
    )

    w_start = max(
        0,
        (src_w - target_width) // 2
    )

    d_end = min(
        src_d,
        d_start + target_depth
    )

    h_end = min(
        src_h,
        h_start + target_height
    )

    w_end = min(
        src_w,
        w_start + target_width
    )

    cropped = source[
        d_start:d_end,
        h_start:h_end,
        w_start:w_end
    ]

    # --------------------------------------------------------
    # Destination placement
    # --------------------------------------------------------

    out_d_start = max(
        0,
        (target_depth - cropped.shape[0]) // 2
    )

    out_h_start = max(
        0,
        (target_height - cropped.shape[1]) // 2
    )

    out_w_start = max(
        0,
        (target_width - cropped.shape[2]) // 2
    )

    out_d_end = (
        out_d_start
        + cropped.shape[0]
    )

    out_h_end = (
        out_h_start
        + cropped.shape[1]
    )

    out_w_end = (
        out_w_start
        + cropped.shape[2]
    )

    output[
        out_d_start:out_d_end,
        out_h_start:out_h_end,
        out_w_start:out_w_end
    ] = cropped

    return output


# ============================================================
# MRI PREPROCESSING
# ============================================================

def preprocess_mri(
    volume
):

    normalized = normalize_volume(
        volume
    )

    processed = center_crop_or_pad_3d(
        normalized,
        TARGET_DEPTH,
        TARGET_HEIGHT,
        TARGET_WIDTH
    )

    # [D,H,W] -> [C,D,H,W]
    tensor = torch.from_numpy(
        processed
    ).unsqueeze(0)

    return tensor.float()


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(
    row
):

    labels = np.zeros(
        len(TARGET_COLUMNS),
        dtype=np.int64
    )

    mask = np.zeros(
        len(TARGET_COLUMNS),
        dtype=np.float32
    )

    missing_count = 0

    for index, column in enumerate(
        TARGET_COLUMNS
    ):

        value = row[column]

        if pd.isna(value):

            missing_count += 1

            continue

        value = str(
            value
        ).strip()

        if value not in SEVERITY_TO_INDEX:

            raise ValueError(
                f"Unknown label '{value}' "
                f"for {column}"
            )

        labels[index] = (
            SEVERITY_TO_INDEX[value]
        )

        mask[index] = 1.0

    return (
        torch.from_numpy(labels),
        torch.from_numpy(mask),
        missing_count,
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

class RSNALumbarClassificationDataset(
    Dataset
):

    def __init__(
        self,
        manifest,
        cache=False
    ):

        self.manifest = (
            manifest
            .reset_index(drop=True)
            .copy()
        )

        self.cache_enabled = cache

        self.cache = {}

    def __len__(self):

        return len(
            self.manifest
        )

    def __getitem__(
        self,
        index
    ):

        if (
            self.cache_enabled
            and index in self.cache
        ):

            return self.cache[index]

        row = self.manifest.iloc[
            index
        ]

        study_id = str(
            row["study_id"]
        )

        series_id = str(
            row["series_id"]
        )

        series_description = str(
            row["series_description"]
        )

        series_path = Path(
            str(row["series_path"])
        )

        # ----------------------------------------------------
        # Load MRI
        # ----------------------------------------------------

        volume = load_dicom_series(
            series_path
        )

        image = preprocess_mri(
            volume
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        (
            labels,
            label_mask,
            missing_count,
        ) = encode_labels(
            row
        )

        sample = {
            "image": image,

            "labels": labels,

            "label_mask":
                label_mask,

            "study_id":
                study_id,

            "series_id":
                series_id,

            "series_description":
                series_description,

            "original_shape":
                torch.tensor(
                    volume.shape,
                    dtype=torch.int64
                ),

            "missing_label_count":
                torch.tensor(
                    missing_count,
                    dtype=torch.int64
                ),
        }

        if self.cache_enabled:

            self.cache[index] = sample

        return sample


# ============================================================
# CUSTOM COLLATE FUNCTION
# ============================================================

def classification_collate(
    batch
):

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ],
        dim=0
    )

    labels = torch.stack(
        [
            item["labels"]
            for item in batch
        ],
        dim=0
    )

    label_mask = torch.stack(
        [
            item["label_mask"]
            for item in batch
        ],
        dim=0
    )

    study_ids = [
        item["study_id"]
        for item in batch
    ]

    series_ids = [
        item["series_id"]
        for item in batch
    ]

    series_descriptions = [
        item["series_description"]
        for item in batch
    ]

    original_shapes = torch.stack(
        [
            item["original_shape"]
            for item in batch
        ],
        dim=0
    )

    missing_label_counts = torch.stack(
        [
            item["missing_label_count"]
            for item in batch
        ],
        dim=0
    )

    return {
        "images": images,

        "labels": labels,

        "label_mask": label_mask,

        "study_ids": study_ids,

        "series_ids": series_ids,

        "series_descriptions":
            series_descriptions,

        "original_shapes":
            original_shapes,

        "missing_label_counts":
            missing_label_counts,
    }


# ============================================================
# DATASET CREATION
# ============================================================

def create_datasets(
    train_manifest,
    val_manifest
):

    banner("CREATING PYTORCH DATASETS")

    train_dataset = (
        RSNALumbarClassificationDataset(
            train_manifest,
            cache=False
        )
    )

    val_dataset = (
        RSNALumbarClassificationDataset(
            val_manifest,
            cache=False
        )
    )

    print(
        f"Training dataset length   : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation dataset length : "
        f"{len(val_dataset)}"
    )

    return (
        train_dataset,
        val_dataset
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset
):

    banner("CREATING PYTORCH DATALOADERS")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=classification_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=classification_collate,
    )

    print(
        f"Batch size       : {BATCH_SIZE}"
    )

    print(
        f"Workers          : {NUM_WORKERS}"
    )

    print(
        f"Training batches : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# BATCH VALIDATION
# ============================================================

def validate_batch(
    batch,
    batch_number,
    split
):

    images = batch["images"]

    labels = batch["labels"]

    label_mask = batch[
        "label_mask"
    ]

    expected_image_shape = (
        BATCH_SIZE,
        1,
        TARGET_DEPTH,
        TARGET_HEIGHT,
        TARGET_WIDTH,
    )

    expected_label_shape = (
        BATCH_SIZE,
        len(TARGET_COLUMNS)
    )

    # Last batch may contain fewer samples.
    actual_batch_size = (
        images.shape[0]
    )

    expected_image_shape = (
        actual_batch_size,
        1,
        TARGET_DEPTH,
        TARGET_HEIGHT,
        TARGET_WIDTH,
    )

    expected_label_shape = (
        actual_batch_size,
        len(TARGET_COLUMNS)
    )

    if tuple(images.shape) != (
        expected_image_shape
    ):

        raise RuntimeError(
            f"Unexpected image shape: "
            f"{tuple(images.shape)}"
        )

    if tuple(labels.shape) != (
        expected_label_shape
    ):

        raise RuntimeError(
            f"Unexpected label shape: "
            f"{tuple(labels.shape)}"
        )

    if tuple(label_mask.shape) != (
        expected_label_shape
    ):

        raise RuntimeError(
            f"Unexpected label-mask shape: "
            f"{tuple(label_mask.shape)}"
        )

    if not torch.isfinite(
        images
    ).all():

        raise RuntimeError(
            "Image tensor contains "
            "non-finite values."
        )

    if images.min().item() < -1e-5:

        raise RuntimeError(
            "Image tensor has values below 0."
        )

    if images.max().item() > 1.00001:

        raise RuntimeError(
            "Image tensor has values above 1."
        )

    if not torch.isfinite(
        labels.float()
    ).all():

        raise RuntimeError(
            "Labels contain non-finite values."
        )

    if not torch.isfinite(
        label_mask
    ).all():

        raise RuntimeError(
            "Label mask contains non-finite values."
        )

    # Valid labels must be 0,1,2.
    observed_labels = labels[
        label_mask > 0
    ]

    if observed_labels.numel() > 0:

        if (
            observed_labels.min().item()
            < 0
            or
            observed_labels.max().item()
            > 2
        ):

            raise RuntimeError(
                "Invalid severity index."
            )

    # Mask must be binary.
    unique_mask = torch.unique(
        label_mask
    )

    for value in unique_mask.tolist():

        if value not in (
            0.0,
            1.0,
        ):

            raise RuntimeError(
                "Label mask is not binary."
            )

    return {
        "split": split,

        "batch_number":
            batch_number,

        "batch_size":
            actual_batch_size,

        "image_shape":
            str(tuple(images.shape)),

        "label_shape":
            str(tuple(labels.shape)),

        "label_mask_shape":
            str(tuple(label_mask.shape)),

        "image_min":
            float(images.min().item()),

        "image_max":
            float(images.max().item()),

        "image_mean":
            float(images.mean().item()),

        "image_std":
            float(images.std().item()),

        "labeled_targets":
            int(
                label_mask.sum().item()
            ),

        "missing_targets":
            int(
                (
                    label_mask == 0
                ).sum().item()
            ),

        "study_ids":
            "|".join(
                batch["study_ids"]
            ),

        "status":
            "PASS",
    }


# ============================================================
# VALIDATE DATALOADER
# ============================================================

def validate_dataloader(
    loader,
    split,
    max_batches
):

    results = []

    banner(
        f"VALIDATING {split.upper()} DATALOADER"
    )

    iterator = iter(
        loader
    )

    for batch_number in range(
        1,
        max_batches + 1
    ):

        try:

            batch = next(
                iterator
            )

        except StopIteration:

            break

        print(
            f"{split.upper()} batch "
            f"{batch_number}/{max_batches}"
        )

        print(
            f"    images : "
            f"{tuple(batch['images'].shape)}"
        )

        print(
            f"    labels : "
            f"{tuple(batch['labels'].shape)}"
        )

        print(
            f"    mask   : "
            f"{tuple(batch['label_mask'].shape)}"
        )

        print(
            f"    studies: "
            f"{batch['study_ids']}"
        )

        result = validate_batch(
            batch,
            batch_number,
            split
        )

        results.append(
            result
        )

        print(
            f"    status : PASS"
        )

    return results


# ============================================================
# STUDY LEAKAGE CHECK
# ============================================================

def check_study_leakage(
    train_manifest,
    val_manifest
):

    train_ids = set(
        train_manifest[
            "study_id"
        ].astype(str)
    )

    val_ids = set(
        val_manifest[
            "study_id"
        ].astype(str)
    )

    overlap = (
        train_ids.intersection(
            val_ids
        )
    )

    return sorted(
        overlap
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def calculate_label_statistics(
    train_manifest,
    val_manifest
):

    statistics = []

    for split_name, manifest in [
        ("train", train_manifest),
        ("validation", val_manifest),
    ]:

        for target in TARGET_COLUMNS:

            counts = (
                manifest[target]
                .value_counts(
                    dropna=False
                )
            )

            normal = int(
                counts.get(
                    "Normal/Mild",
                    0
                )
            )

            moderate = int(
                counts.get(
                    "Moderate",
                    0
                )
            )

            severe = int(
                counts.get(
                    "Severe",
                    0
                )
            )

            missing = int(
                manifest[target]
                .isna()
                .sum()
            )

            statistics.append(
                {
                    "split":
                        split_name,

                    "target":
                        target,

                    "normal_mild":
                        normal,

                    "moderate":
                        moderate,

                    "severe":
                        severe,

                    "missing":
                        missing,

                    "total":
                        len(manifest),
                }
            )

    return pd.DataFrame(
        statistics
    )


# ============================================================
# WRITE REPORT
# ============================================================

def write_report(
    train_manifest,
    val_manifest,
    batch_results,
    overlap,
    label_statistics
):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "PART 89 — RSNA CLASSIFICATION DATASET PIPELINE"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        f"Training studies: "
        f"{len(train_manifest)}"
    )

    lines.append(
        f"Validation studies: "
        f"{len(val_manifest)}"
    )

    lines.append(
        f"Study overlap: "
        f"{len(overlap)}"
    )

    lines.append("")

    lines.append(
        "DATASET CONFIGURATION"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Batch size: {BATCH_SIZE}"
    )

    lines.append(
        f"Workers: {NUM_WORKERS}"
    )

    lines.append(
        f"Target depth: {TARGET_DEPTH}"
    )

    lines.append(
        f"Target height: {TARGET_HEIGHT}"
    )

    lines.append(
        f"Target width: {TARGET_WIDTH}"
    )

    lines.append(
        f"Number of classification targets: "
        f"{len(TARGET_COLUMNS)}"
    )

    lines.append("")

    lines.append(
        "LABEL ENCODING"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "0 = Normal/Mild"
    )

    lines.append(
        "1 = Moderate"
    )

    lines.append(
        "2 = Severe"
    )

    lines.append(
        "Missing labels are represented by "
        "label_mask = 0."
    )

    lines.append("")

    lines.append(
        "BATCH VALIDATION"
    )

    lines.append(
        "-" * 78
    )

    passed = int(
        (
            batch_results["status"]
            == "PASS"
        ).sum()
    )

    failed = int(
        (
            batch_results["status"]
            != "PASS"
        ).sum()
    )

    lines.append(
        f"Batches tested: "
        f"{len(batch_results)}"
    )

    lines.append(
        f"Batches passed: {passed}"
    )

    lines.append(
        f"Batches failed: {failed}"
    )

    lines.append("")

    lines.append(
        "STUDY LEAKAGE"
    )

    lines.append(
        "-" * 78
    )

    if overlap:

        lines.append(
            "FAIL — overlapping study IDs detected."
        )

        for study_id in overlap[:20]:

            lines.append(
                f"  {study_id}"
            )

    else:

        lines.append(
            "PASS — no study overlap."
        )

    lines.append("")

    lines.append(
        "LABEL COVERAGE"
    )

    lines.append(
        "-" * 78
    )

    for split_name in [
        "train",
        "validation",
    ]:

        subset = label_statistics[
            label_statistics["split"]
            == split_name
        ]

        total_targets = int(
            subset["total"].sum()
            * 1
        )

        missing_targets = int(
            subset["missing"].sum()
        )

        labeled_targets = (
            total_targets
            - missing_targets
        )

        lines.append(
            f"{split_name}: "
            f"labeled={labeled_targets}, "
            f"missing={missing_targets}"
        )

    lines.append("")

    lines.append(
        "FINAL STATUS"
    )

    lines.append(
        "-" * 78
    )

    if (
        len(overlap) == 0
        and failed == 0
        and len(batch_results) > 0
    ):

        lines.append(
            "PASS — Classification Dataset and "
            "DataLoader pipeline validated."
        )

    else:

        lines.append(
            "REVIEW REQUIRED — Dataset pipeline "
            "validation failed."
        )

    lines.append("")

    lines.append(
        "No classifier training was performed."
    )

    lines.append(
        "Segmentation models/checkpoints were "
        "not modified."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "PART 89 — RSNA CLASSIFICATION DATASET & DATALOADER"
    )

    print(
        f"PyTorch version : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available  : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA device     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    (
        train_manifest,
        val_manifest
    ) = load_manifests()

    # --------------------------------------------------------
    # Study leakage
    # --------------------------------------------------------

    banner(
        "STUDY-LEVEL LEAKAGE CHECK"
    )

    overlap = check_study_leakage(
        train_manifest,
        val_manifest
    )

    print(
        f"Training studies   : "
        f"{len(train_manifest)}"
    )

    print(
        f"Validation studies : "
        f"{len(val_manifest)}"
    )

    print(
        f"Study overlap      : "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "Study-level leakage detected."
        )

    # --------------------------------------------------------
    # Label statistics
    # --------------------------------------------------------

    banner(
        "CLASSIFICATION LABEL STATISTICS"
    )

    label_statistics = (
        calculate_label_statistics(
            train_manifest,
            val_manifest
        )
    )

    for split_name in [
        "train",
        "validation",
    ]:

        subset = label_statistics[
            label_statistics["split"]
            == split_name
        ]

        print(
            f"{split_name.upper()} "
            f"missing target values: "
            f"{int(subset['missing'].sum())}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset
    ) = create_datasets(
        train_manifest,
        val_manifest
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        train_loader,
        val_loader
    ) = create_dataloaders(
        train_dataset,
        val_dataset
    )

    # --------------------------------------------------------
    # Batch validation
    # --------------------------------------------------------

    train_results = validate_dataloader(
        train_loader,
        "train",
        PREFETCH_BATCHES
    )

    val_results = validate_dataloader(
        val_loader,
        "validation",
        PREFETCH_BATCHES
    )

    batch_results = pd.DataFrame(
        train_results
        + val_results
    )

    # --------------------------------------------------------
    # Save batch results
    # --------------------------------------------------------

    batch_results.to_csv(
        BATCH_RESULTS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = write_report(
        train_manifest,
        val_manifest,
        batch_results,
        overlap,
        label_statistics
    )

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    passed_batches = int(
        (
            batch_results["status"]
            == "PASS"
        ).sum()
    )

    failed_batches = int(
        (
            batch_results["status"]
            != "PASS"
        ).sum()
    )

    overall_pass = (
        len(overlap) == 0
        and failed_batches == 0
        and len(batch_results) > 0
    )

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    summary = {

        "part": 89,

        "title":
            "RSNA Classification Dataset and DataLoader Pipeline",

        "seed":
            SEED,

        "dataset": {

            "training_studies":
                int(len(train_manifest)),

            "validation_studies":
                int(len(val_manifest)),

            "classification_targets":
                int(len(TARGET_COLUMNS)),
        },

        "dataloader": {

            "batch_size":
                BATCH_SIZE,

            "num_workers":
                NUM_WORKERS,

            "training_batches":
                int(len(train_loader)),

            "validation_batches":
                int(len(val_loader)),
        },

        "preprocessing": {

            "target_shape":
                [
                    1,
                    TARGET_DEPTH,
                    TARGET_HEIGHT,
                    TARGET_WIDTH,
                ],

            "normalization":
                "1st-99th percentile clipping",

            "normalization_range":
                [
                    0.0,
                    1.0,
                ],
        },

        "label_encoding": {

            "Normal/Mild":
                0,

            "Moderate":
                1,

            "Severe":
                2,

            "missing_label_representation":
                "label_mask=0",
        },

        "study_leakage": {

            "overlap_count":
                int(len(overlap)),

            "passed":
                len(overlap) == 0,
        },

        "batch_validation": {

            "batches_tested":
                int(len(batch_results)),

            "batches_passed":
                passed_batches,

            "batches_failed":
                failed_batches,
        },

        "outputs": {

            "batch_results":
                str(BATCH_RESULTS_CSV),

            "report":
                str(REPORT_TXT),

            "summary_json":
                str(SUMMARY_JSON),
        },

        "training_performed":
            False,

        "segmentation_modified":
            False,

        "overall_status":
            "PASS"
            if overall_pass
            else "REVIEW_REQUIRED",
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    banner(
        "PART 89 COMPLETE"
    )

    print(
        f"Training studies    : "
        f"{len(train_manifest)}"
    )

    print(
        f"Validation studies  : "
        f"{len(val_manifest)}"
    )

    print(
        f"Study overlap       : "
        f"{len(overlap)}"
    )

    print(
        f"Batches tested      : "
        f"{len(batch_results)}"
    )

    print(
        f"Batches passed      : "
        f"{passed_batches}"
    )

    print(
        f"Batches failed      : "
        f"{failed_batches}"
    )

    print(
        f"Image tensor        : "
        f"(B, 1, {TARGET_DEPTH}, "
        f"{TARGET_HEIGHT}, {TARGET_WIDTH})"
    )

    print(
        f"Label tensor        : "
        f"(B, {len(TARGET_COLUMNS)})"
    )

    print(
        f"Label mask          : "
        f"(B, {len(TARGET_COLUMNS)})"
    )

    print(
        f"\nBatch results:\n"
        f"{BATCH_RESULTS_CSV}"
    )

    print(
        f"\nReport:\n"
        f"{REPORT_TXT}"
    )

    print(
        f"\nSummary JSON:\n"
        f"{SUMMARY_JSON}"
    )

    if overall_pass:

        print(
            "\nSTATUS: PASS — "
            "CLASSIFICATION DATASET PIPELINE VALIDATED"
        )

    else:

        print(
            "\nSTATUS: REVIEW REQUIRED — "
            "DO NOT START CLASSIFIER TRAINING"
        )


if __name__ == "__main__":
    main()