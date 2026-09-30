from pathlib import Path
import sys
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "src"))

import segmentation_rsna_part9_3d_dataset_loader as p9
import segmentation_rsna_part11_controlled_pilot_training as p11


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 6

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

CROP_SIZE = (32, 64, 64)

VAL_CASES = 50

CHECKPOINT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part84_final_reproducible_training"
    / "checkpoints"
    / "part84_best_model.pth"
)

VAL_COHORT = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part15_extended_controlled_training"
    / "part15_validation_cohort.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part85_final_segmentation_audit"
)

VIS_DIR = OUTPUT_DIR / "visualizations"

CASE_CSV = OUTPUT_DIR / "part85_case_metrics.csv"

REPORT_TXT = (
    ROOT
    / "reports"
    / "part85_final_segmentation_audit_report.txt"
)

SUMMARY_JSON = (
    ROOT
    / "reports"
    / "part85_final_segmentation_audit_summary.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CENTERED CROP
# ============================================================

def centered_crop_3d(image, mask, crop_size):
    """
    Center crop a 3D image and corresponding mask.

    image: [D,H,W]
    mask : [D,H,W]
    """

    cd, ch, cw = crop_size

    d, h, w = image.shape

    if d < cd or h < ch or w < cw:
        raise ValueError(
            f"Input shape {image.shape} is smaller than crop {crop_size}"
        )

    sd = (d - cd) // 2
    sh = (h - ch) // 2
    sw = (w - cw) // 2

    image_crop = image[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw
    ]

    mask_crop = mask[
        sd:sd + cd,
        sh:sh + ch,
        sw:sw + cw
    ]

    return image_crop, mask_crop


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print("\nLoading SwinUNETR model...")

    model = p11.create_model(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint compatibility failure.\n"
            f"Missing keys: {missing[:20]}\n"
            f"Unexpected keys: {unexpected[:20]}"
        )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print(f"Checkpoint: {CHECKPOINT}")

    return model


# ============================================================
# DICE / PRECISION / RECALL
# ============================================================

def binary_metrics(pred, target):
    """
    pred and target are boolean tensors.
    """

    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, ~target).sum().item()
    fn = torch.logical_and(~pred, target).sum().item()

    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / denominator

    precision_den = tp + fp

    if precision_den == 0:
        precision = 0.0
    else:
        precision = tp / precision_den

    recall_den = tp + fn

    if recall_den == 0:
        recall = 0.0
    else:
        recall = tp / recall_den

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    case_index,
    image,
    target,
    prediction,
    case_name
):
    """
    Saves a representative middle-slice visualization.

    Four panels:
    1. MRI
    2. Pseudo-mask
    3. Prediction
    4. Prediction overlay
    """

    D = image.shape[0]

    # Select middle slice.
    z = D // 2

    img_slice = image[z]
    target_slice = target[z]
    pred_slice = prediction[z]

    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img_slice, cmap="gray")
    ax1.set_title("MRI")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(target_slice, interpolation="nearest")
    ax2.set_title("Pseudo-mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(pred_slice, interpolation="nearest")
    ax3.set_title("Prediction")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(img_slice, cmap="gray")

    overlay = np.ma.masked_where(
        pred_slice == 0,
        pred_slice
    )

    ax4.imshow(
        overlay,
        alpha=0.55,
        interpolation="nearest"
    )

    ax4.set_title("Prediction Overlay")
    ax4.axis("off")

    fig.suptitle(
        f"Part 85 | Case {case_index + 1} | {case_name}"
    )

    plt.tight_layout()

    filename = (
        VIS_DIR
        / f"case_{case_index + 1:03d}_{case_name}.png"
    )

    fig.savefig(
        filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return filename


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_TXT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 75)
    print("PART 85 — FINAL SEGMENTATION AUDIT")
    print("=" * 75)

    print(f"Project root : {ROOT}")
    print(f"Device       : {DEVICE}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Validation   : {VAL_COHORT}")
    print(f"Validation cases: {VAL_CASES}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Part 84 checkpoint not found:\n{CHECKPOINT}"
        )

    if not VAL_COHORT.exists():
        raise FileNotFoundError(
            f"Validation cohort not found:\n{VAL_COHORT}"
        )

    # --------------------------------------------------------
    # Load validation cohort
    # --------------------------------------------------------

    import pandas as pd

    val_df = pd.read_csv(VAL_COHORT)

    if len(val_df) < VAL_CASES:
        raise RuntimeError(
            f"Validation cohort contains only {len(val_df)} rows."
        )

    val_df = val_df.iloc[:VAL_CASES].copy()

    print(f"\nLoaded validation rows: {len(val_df)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Global accumulators
    # --------------------------------------------------------

    global_tp = np.zeros(NUM_CLASSES, dtype=np.float64)
    global_fp = np.zeros(NUM_CLASSES, dtype=np.float64)
    global_fn = np.zeros(NUM_CLASSES, dtype=np.float64)

    case_fg_dice = []
    case_macro_dice = []

    class_case_dice = {
        c: []
        for c in range(1, NUM_CLASSES)
    }

    rows = []

    visualization_indices = {
        0,
        9,
        19,
        29,
        39,
        49,
    }

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for idx, (_, row) in enumerate(val_df.iterrows()):

            print(
                f"\n[{idx + 1:02d}/{VAL_CASES}] "
                "Loading case..."
            )

            loaded = p11.load_tensor_case(
                row,
                p9
            )

            image = loaded[0]
            mask = loaded[1]

            # ------------------------------------------------
            # Normalize image dimensionality
            # ------------------------------------------------

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            image = image.float()
            mask = mask.long()

            # Image can be [1,D,H,W].
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)

            # Mask can also occasionally have a singleton channel.
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            if image.ndim != 3:
                raise RuntimeError(
                    f"Unexpected image shape: {tuple(image.shape)}"
                )

            if mask.ndim != 3:
                raise RuntimeError(
                    f"Unexpected mask shape: {tuple(mask.shape)}"
                )

            # ------------------------------------------------
            # Center crop
            # ------------------------------------------------

            image_crop, mask_crop = centered_crop_3d(
                image,
                mask,
                CROP_SIZE
            )

            # ------------------------------------------------
            # Model input
            # ------------------------------------------------

            x = image_crop.unsqueeze(0).unsqueeze(0).to(DEVICE)

            logits = model(x)

            prediction = torch.argmax(
                logits,
                dim=1
            )[0].cpu()

            target = mask_crop.cpu()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            case_class_dice = []

            case_metrics = {
                "case_index": idx + 1,
            }

            # Try common identifiers.
            possible_names = [
                "study_id",
                "series_id",
                "case_id",
                "image_id",
            ]

            case_name = None

            for col in possible_names:
                if col in row.index:
                    value = str(row[col])

                    if value and value.lower() != "nan":
                        case_name = value
                        break

            if case_name is None:
                case_name = f"case_{idx + 1:03d}"

            case_metrics["case_name"] = case_name

            total_target_fg = 0
            total_pred_fg = 0

            for c in range(1, NUM_CLASSES):

                pred_c = prediction == c
                target_c = target == c

                metrics = binary_metrics(
                    pred_c,
                    target_c
                )

                global_tp[c] += metrics["tp"]
                global_fp[c] += metrics["fp"]
                global_fn[c] += metrics["fn"]

                case_class_dice.append(
                    metrics["dice"]
                )

                class_case_dice[c].append(
                    metrics["dice"]
                )

                case_metrics[
                    f"class_{c}_dice"
                ] = metrics["dice"]

                case_metrics[
                    f"class_{c}_precision"
                ] = metrics["precision"]

                case_metrics[
                    f"class_{c}_recall"
                ] = metrics["recall"]

                total_target_fg += int(
                    target_c.sum().item()
                )

                total_pred_fg += int(
                    pred_c.sum().item()
                )

            # ------------------------------------------------
            # Case-level aggregate metrics
            # ------------------------------------------------

            fg_dice = np.mean(
                case_class_dice
            )

            case_macro = np.mean(
                [binary_metrics(
                    prediction == c,
                    target == c
                )["dice"] for c in range(1, NUM_CLASSES)]
            )

            case_fg_dice.append(
                fg_dice
            )

            case_macro_dice.append(
                case_macro
            )

            case_metrics[
                "foreground_dice"
            ] = float(fg_dice)

            case_metrics[
                "macro_foreground_dice"
            ] = float(case_macro)

            case_metrics[
                "target_foreground_voxels"
            ] = int(
                (target > 0).sum().item()
            )

            case_metrics[
                "predicted_foreground_voxels"
            ] = int(
                (prediction > 0).sum().item()
            )

            rows.append(case_metrics)

            print(
                f"  Case FG Dice       : {fg_dice:.6f}"
            )
            print(
                f"  Target FG voxels   : "
                f"{int((target > 0).sum())}"
            )
            print(
                f"  Predicted FG voxels: "
                f"{int((prediction > 0).sum())}"
            )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            if idx in visualization_indices:

                save_visualization(
                    idx,
                    image_crop.numpy(),
                    target.numpy(),
                    prediction.numpy(),
                    case_name.replace(
                        "/",
                        "_"
                    )
                )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    global_class_dice = {}
    global_class_precision = {}
    global_class_recall = {}

    for c in range(1, NUM_CLASSES):

        tp = global_tp[c]
        fp = global_fp[c]
        fn = global_fn[c]

        denom = (
            2 * tp +
            fp +
            fn
        )

        if denom == 0:
            dice = 1.0
        else:
            dice = (
                2 * tp
                / denom
            )

        precision_den = tp + fp

        if precision_den == 0:
            precision = 0.0
        else:
            precision = (
                tp
                / precision_den
            )

        recall_den = tp + fn

        if recall_den == 0:
            recall = 0.0
        else:
            recall = (
                tp
                / recall_den
            )

        global_class_dice[c] = float(dice)
        global_class_precision[c] = float(precision)
        global_class_recall[c] = float(recall)

    global_fg_dice = float(
        np.mean(
            list(global_class_dice.values())
        )
    )

    mean_case_fg_dice = float(
        np.mean(case_fg_dice)
    )

    median_case_fg_dice = float(
        np.median(case_fg_dice)
    )

    macro_fg_dice = float(
        np.mean(
            [
                np.mean(class_case_dice[c])
                for c in range(1, NUM_CLASSES)
            ]
        )
    )

    # ========================================================
    # SAVE CASE CSV
    # ========================================================

    if rows:

        fieldnames = list(rows[0].keys())

        with open(
            CASE_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "part": 85,
        "title": "Final Segmentation Audit",
        "checkpoint": str(CHECKPOINT),
        "validation_cohort": str(VAL_COHORT),
        "validation_cases": VAL_CASES,
        "crop_size": list(CROP_SIZE),
        "num_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "seed": SEED,

        "global_foreground_dice": global_fg_dice,
        "mean_case_foreground_dice": mean_case_fg_dice,
        "median_case_foreground_dice": median_case_fg_dice,
        "macro_foreground_dice": macro_fg_dice,

        "classwise": {}
    }

    for c in range(1, NUM_CLASSES):

        summary["classwise"][
            CLASS_NAMES[c]
        ] = {
            "dice": global_class_dice[c],
            "precision": global_class_precision[c],
            "recall": global_class_recall[c],
            "mean_case_dice": float(
                np.mean(
                    class_case_dice[c]
                )
            )
        }

    # --------------------------------------------------------
    # Compare with Part84 verified reference
    # --------------------------------------------------------

    PART84_GLOBAL_DICE = 0.004313
    PART84_MEAN_CASE_DICE = 0.006432

    summary["part84_reference"] = {
        "global_foreground_dice": PART84_GLOBAL_DICE,
        "mean_case_foreground_dice": PART84_MEAN_CASE_DICE,
        "global_difference":
            global_fg_dice - PART84_GLOBAL_DICE,
        "mean_case_difference":
            mean_case_fg_dice - PART84_MEAN_CASE_DICE,
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

    # ========================================================
    # TEXT REPORT
    # ========================================================

    lines = []

    lines.append("=" * 75)
    lines.append(
        "PART 85 — FINAL SEGMENTATION AUDIT REPORT"
    )
    lines.append("=" * 75)
    lines.append("")

    lines.append(
        f"Checkpoint: {CHECKPOINT}"
    )

    lines.append(
        f"Validation cohort: {VAL_COHORT}"
    )

    lines.append(
        f"Validation cases: {VAL_CASES}"
    )

    lines.append(
        f"Crop size: {CROP_SIZE}"
    )

    lines.append(
        f"Device: {DEVICE}"
    )

    lines.append("")

    lines.append(
        "OVERALL METRICS"
    )

    lines.append("-" * 75)

    lines.append(
        f"Global foreground Dice : "
        f"{global_fg_dice:.6f}"
    )

    lines.append(
        f"Mean-case foreground Dice : "
        f"{mean_case_fg_dice:.6f}"
    )

    lines.append(
        f"Median-case foreground Dice : "
        f"{median_case_fg_dice:.6f}"
    )

    lines.append(
        f"Macro foreground Dice : "
        f"{macro_fg_dice:.6f}"
    )

    lines.append("")

    lines.append(
        "CLASSWISE METRICS"
    )

    lines.append("-" * 75)

    for c in range(1, NUM_CLASSES):

        lines.append(
            f"{c}. {CLASS_NAMES[c]}"
        )

        lines.append(
            f"   Dice      : "
            f"{global_class_dice[c]:.6f}"
        )

        lines.append(
            f"   Precision : "
            f"{global_class_precision[c]:.6f}"
        )

        lines.append(
            f"   Recall    : "
            f"{global_class_recall[c]:.6f}"
        )

    lines.append("")

    lines.append(
        "PART 84 REFERENCE"
    )

    lines.append("-" * 75)

    lines.append(
        f"Part84 global Dice : "
        f"{PART84_GLOBAL_DICE:.6f}"
    )

    lines.append(
        f"Part85 global Dice : "
        f"{global_fg_dice:.6f}"
    )

    lines.append(
        f"Difference : "
        f"{global_fg_dice - PART84_GLOBAL_DICE:+.6f}"
    )

    lines.append("")

    lines.append(
        "QUALITATIVE OUTPUTS"
    )

    lines.append("-" * 75)

    lines.append(
        f"Visualization directory: {VIS_DIR}"
    )

    lines.append(
        f"Case metrics CSV: {CASE_CSV}"
    )

    lines.append("")

    lines.append(
        "FINAL INTERPRETATION"
    )

    lines.append("-" * 75)

    lines.append(
        "Part 85 is a frozen audit of the Part 84 "
        "reproducible segmentation checkpoint."
    )

    lines.append(
        "No model training or parameter updates were "
        "performed during this audit."
    )

    if global_fg_dice >= PART84_GLOBAL_DICE * 0.95:
        lines.append(
            "The Part 85 global Dice is consistent with "
            "the verified Part 84 reference."
        )
    else:
        lines.append(
            "The Part 85 global Dice differs substantially "
            "from the Part 84 reference and should be investigated."
        )

    lines.append("")

    report_text = "\n".join(lines)

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 75)
    print("PART 85 COMPLETE")
    print("=" * 75)

    print(
        f"Global FG Dice       : {global_fg_dice:.6f}"
    )

    print(
        f"Mean-case FG Dice    : {mean_case_fg_dice:.6f}"
    )

    print(
        f"Median-case FG Dice  : {median_case_fg_dice:.6f}"
    )

    print(
        f"Macro FG Dice        : {macro_fg_dice:.6f}"
    )

    print("\nClasswise:")

    for c in range(1, NUM_CLASSES):

        print(
            f"  C{c} "
            f"{CLASS_NAMES[c]:45s} "
            f"Dice={global_class_dice[c]:.6f} "
            f"P={global_class_precision[c]:.6f} "
            f"R={global_class_recall[c]:.6f}"
        )

    print("\nOutputs:")

    print(
        f"  Case CSV       : {CASE_CSV}"
    )

    print(
        f"  Visualizations : {VIS_DIR}"
    )

    print(
        f"  TXT report     : {REPORT_TXT}"
    )

    print(
        f"  JSON summary   : {SUMMARY_JSON}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()