
"""
PART 46 — CHECKPOINT TRANSITION FORENSICS

Purpose
-------
Analyze the existing Part 43/44 checkpoints around the foreground-collapse
transition. This is NOT a training run.

Questions:
1. When does foreground argmax collapse?
2. Does background probability rise, or does foreground probability merely
   become insufficient to win argmax?
3. How do foreground/background logits change?
4. What happens to output gradients at the saved checkpoints?
5. How much do model parameters change between consecutive checkpoints?
6. Does the collapse appear abrupt or gradual?

Important:
- No optimizer steps.
- No training.
- Part 15 is untouched.
- Part 43/44 checkpoints are read-only.
- No SPIDER.
- No test set.
"""

from pathlib import Path
import sys, json, csv, hashlib, importlib.util
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss

ROOT = Path(r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project")
SRC = ROOT / "src"

PART11_PATH = SRC / "segmentation_rsna_part11_controlled_pilot_training.py"
PART9_PATH = SRC / "segmentation_rsna_part9_3d_dataset_loader.py"

P15 = ROOT / "outputs" / "segmentation" / "rsna_part15_extended_controlled_training"
INIT = P15 / "checkpoints" / "part15_initialization_from_part11.pth"
VACSV = P15 / "part15_validation_cohort.csv"

PART43 = ROOT / "outputs" / "segmentation" / "rsna_part43_improved_spatial_sampling_training"
P43R = PART43 / "random_spatial_crop"
P43C = PART43 / "foreground_centered_spatial_crop"

PART44 = ROOT / "outputs" / "segmentation" / "rsna_part44_loss_component_class_imbalance_diagnostic"
P44O = PART44 / "original_dicece"
P44A = PART44 / "foreground_aware_ce"

OUT = ROOT / "outputs" / "segmentation" / "rsna_part46_checkpoint_transition_forensics"
REPORT = OUT / "reports"
REPORT.mkdir(parents=True, exist_ok=True)

VAL_N = 20
FULL_SHAPE = (64, 96, 96)
CROP_SHAPE = (32, 64, 64)
SEED_OFFSET = 5000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dice_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,
    lambda_ce=0.0,
)
ce_fn = torch.nn.CrossEntropyLoss()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def centered_crop(image, mask, shape, seed):
    D, H, W = image.shape
    cd, ch, cw = shape
    rng = np.random.default_rng(seed)
    fg = torch.nonzero(mask > 0, as_tuple=False)

    if len(fg):
        p = fg[int(rng.integers(0, len(fg)))]
        center = [
            int(p[0]) + int(rng.integers(-cd // 8, cd // 8 + 1)),
            int(p[1]) + int(rng.integers(-ch // 8, ch // 8 + 1)),
            int(p[2]) + int(rng.integers(-cw // 8, cw // 8 + 1)),
        ]
    else:
        center = [D // 2, H // 2, W // 2]

    starts = [
        max(0, min(D - cd, center[0] - cd // 2)),
        max(0, min(H - ch, center[1] - ch // 2)),
        max(0, min(W - cw, center[2] - cw // 2)),
    ]

    d, h, w = starts
    return (
        image[d:d+cd, h:h+ch, w:w+cw],
        mask[d:d+cd, h:h+ch, w:w+cw],
    )


def load_cases(part11, part9):
    df = pd.read_csv(VACSV)
    cases = []

    for i in range(min(VAL_N, len(df))):
        row = df.iloc[i]
        z = part11.load_tensor_case(row, part9)

        image = torch.as_tensor(z[0])
        mask = torch.as_tensor(z[1])

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        image = image.float()
        mask = mask.long()

        if tuple(image.shape) != FULL_SHAPE:
            raise RuntimeError(f"Unexpected image shape {tuple(image.shape)}")
        if tuple(mask.shape) != FULL_SHAPE:
            raise RuntimeError(f"Unexpected mask shape {tuple(mask.shape)}")

        image, mask = centered_crop(
            image,
            mask,
            CROP_SHAPE,
            SEED_OFFSET + i,
        )

        cases.append((image, mask))

        if i == 0 or (i + 1) % 10 == 0 or i + 1 == min(VAL_N, len(df)):
            print(
                f"Validation case {i+1:03d}/{min(VAL_N,len(df))} "
                f"cropFG={(mask > 0).sum().item()}"
            )

    print(
        f"Mean validation crop FG : "
        f"{np.mean([(m > 0).sum().item() for _, m in cases]):.2f}"
    )

    return cases


def make_model(part11):
    model = part11.create_model(DEVICE)
    return model


def extract_state(obj):
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
    return obj


def load_checkpoint(model, path):
    state = torch.load(path, map_location=DEVICE)
    state = extract_state(state)
    model.load_state_dict(state, strict=True)


def loss_components(logits, target):
    y = target.unsqueeze(1)
    dice = dice_fn(logits, y)
    ce = ce_fn(logits, target)
    return dice + ce, dice, ce


def model_parameter_stats(model):
    total_sq = 0.0
    total_abs = 0.0
    n = 0

    for p in model.parameters():
        x = p.detach().float()
        total_sq += float((x * x).sum().item())
        total_abs += float(x.abs().sum().item())
        n += x.numel()

    return {
        "parameter_l2": float(np.sqrt(total_sq)),
        "parameter_mean_abs": float(total_abs / max(1, n)),
        "parameter_count": int(n),
    }


def parameter_distance(model_a, model_b):
    sq = 0.0
    abs_sum = 0.0
    n = 0

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        d = pa.detach().float() - pb.detach().float()
        sq += float((d * d).sum().item())
        abs_sum += float(d.abs().sum().item())
        n += d.numel()

    return {
        "parameter_delta_l2": float(np.sqrt(sq)),
        "parameter_delta_mean_abs": float(abs_sum / max(1, n)),
    }


def probe_checkpoint(model, cases):
    model.eval()

    records = []

    for image, target in cases:
        x = image[None, None].to(DEVICE)
        y = target[None].to(DEVICE)

        model.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(x)

        logits.retain_grad()
        total, dice, ce = loss_components(logits, y)
        total.backward()

        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            target_fg = y > 0
            pred_fg = pred > 0

            tp = (target_fg & pred_fg).sum().item()
            pf = pred_fg.sum().item()
            tf = target_fg.sum().item()

            fg_dice = (
                2.0 * tp / (pf + tf)
                if (pf + tf) > 0
                else 1.0
            )

            fg_prob = prob[:, 1:].sum(dim=1)

            fg_mask = target_fg
            bg_mask = ~target_fg

            mean_fg_prob = (
                fg_prob[fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            mean_bg_prob = (
                fg_prob[bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

            bg_prob = prob[:, 0]

            mean_bg_probability = (
                bg_prob[bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

            mean_bg_probability_on_target_fg = (
                bg_prob[fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            foreground_logits = logits[:, 1:].mean(dim=1)
            background_logit = logits[:, 0]

            fg_logit_on_fg = (
                foreground_logits[fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            bg_logit_on_fg = (
                background_logit[fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            fg_logit_on_bg = (
                foreground_logits[bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

            bg_logit_on_bg = (
                background_logit[bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

            g = logits.grad.detach().abs()

            fg_channel_grad = g[:, 1:].mean().item()
            bg_channel_grad = g[:, 0].mean().item()

            fg_logit_grad_on_fg = (
                g[:, 1:].mean(dim=1)[fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            bg_logit_grad_on_fg = (
                g[:, 0][fg_mask].mean().item()
                if fg_mask.any()
                else float("nan")
            )

            fg_logit_grad_on_bg = (
                g[:, 1:].mean(dim=1)[bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

            bg_logit_grad_on_bg = (
                g[:, 0][bg_mask].mean().item()
                if bg_mask.any()
                else float("nan")
            )

        records.append({
            "total_loss": float(total.item()),
            "dice_loss": float(dice.item()),
            "ce_loss": float(ce.item()),
            "fg_dice": float(fg_dice),
            "pred_fg": float(pf),
            "target_fg": float(tf),
            "fg_probability": float(fg_prob.mean().item()),
            "fg_probability_on_target_fg": float(mean_fg_prob),
            "fg_probability_on_target_bg": float(mean_bg_prob),
            "background_probability_on_target_bg": float(mean_bg_probability),
            "background_probability_on_target_fg": float(
                mean_bg_probability_on_target_fg
            ),
            "mean_fg_logit_on_target_fg": float(fg_logit_on_fg),
            "mean_bg_logit_on_target_fg": float(bg_logit_on_fg),
            "mean_fg_logit_on_target_bg": float(fg_logit_on_bg),
            "mean_bg_logit_on_target_bg": float(bg_logit_on_bg),
            "fg_bg_logit_gap_on_target_fg": float(
                fg_logit_on_fg - bg_logit_on_fg
            ),
            "fg_bg_logit_gap_on_target_bg": float(
                fg_logit_on_bg - bg_logit_on_bg
            ),
            "output_grad_mean_abs": float(g.mean().item()),
            "foreground_channel_grad_mean_abs": float(fg_channel_grad),
            "background_channel_grad_mean_abs": float(bg_channel_grad),
            "foreground_background_grad_ratio": float(
                fg_channel_grad / (bg_channel_grad + 1e-12)
            ),
            "fg_logit_grad_on_target_fg": float(
                fg_logit_grad_on_fg
            ),
            "bg_logit_grad_on_target_fg": float(
                bg_logit_grad_on_fg
            ),
            "fg_logit_grad_on_target_bg": float(
                fg_logit_grad_on_bg
            ),
            "bg_logit_grad_on_target_bg": float(
                bg_logit_grad_on_bg
            ),
        })

        model.zero_grad(set_to_none=True)

    keys = records[0].keys()
    return {
        k: float(np.nanmean([r[k] for r in records]))
        for k in keys
    }


def checkpoint_candidates(directory):
    found = {}

    if not directory.exists():
        return found

    for p in sorted(directory.glob("*.pth")):
        name = p.name.lower()

        if "initialization" in name:
            found["initialization"] = p
        elif "epoch_01" in name or "epoch1" in name:
            found["epoch_01"] = p
        elif "epoch_02" in name or "epoch2" in name:
            found["epoch_02"] = p
        elif "epoch_03" in name or "epoch3" in name:
            found["epoch_03"] = p
        elif "epoch_04" in name or "epoch4" in name:
            found["epoch_04"] = p
        elif "epoch_05" in name or "epoch5" in name:
            found["epoch_05"] = p
        elif "best_model" in name:
            found["best_model"] = p

    return found


def analyze_condition(label, directory, part11, cases):
    print("\n" + "=" * 82)
    print(label)
    print("=" * 82)

    candidates = checkpoint_candidates(directory)

    if not candidates:
        print(f"No checkpoints found in: {directory}")
        return []

    ordered_names = [
        "initialization",
        "epoch_01",
        "epoch_02",
        "epoch_03",
        "epoch_04",
        "epoch_05",
        "best_model",
    ]

    model = make_model(part11)
    previous_model = None
    previous_name = None
    rows = []

    for name in ordered_names:
        path = candidates.get(name)

        if path is None:
            continue

        load_checkpoint(model, path)

        stats = model_parameter_stats(model)
        probe = probe_checkpoint(model, cases)

        row = {
            "condition": label,
            "checkpoint": name,
            "path": str(path),
            "sha256": sha256_file(path),
            **stats,
            **probe,
        }

        if previous_model is not None:
            row.update(parameter_distance(previous_model, model))
            row["previous_checkpoint"] = previous_name
        else:
            row["parameter_delta_l2"] = float("nan")
            row["parameter_delta_mean_abs"] = float("nan")
            row["previous_checkpoint"] = ""

        rows.append(row)

        print(
            f"{name:<14} "
            f"loss={probe['total_loss']:.6f} "
            f"FGDice={probe['fg_dice']:.6f} "
            f"PredFG={probe['pred_fg']:.1f} "
            f"FGProb={probe['fg_probability']:.6f} "
            f"FG/BGgapFG={probe['fg_bg_logit_gap_on_target_fg']:.6f} "
            f"GradRatio={probe['foreground_background_grad_ratio']:.6f}"
        )

        snapshot = make_model(part11)
        snapshot.load_state_dict(model.state_dict(), strict=True)
        previous_model = snapshot
        previous_name = name

    return rows


def main():
    print("=" * 82)
    print("PART 46 PATH VALIDATION")
    print("=" * 82)

    paths = [
        ("Project root", ROOT),
        ("Part 11", PART11_PATH),
        ("Part 9", PART9_PATH),
        ("Part 15 initialization", INIT),
        ("Part 15 validation cohort", VACSV),
        ("Part 43 random directory", P43R),
        ("Part 43 centered directory", P43C),
        ("Part 44 original directory", P44O),
        ("Part 44 aware directory", P44A),
    ]

    for label, path in paths:
        print(f"{label:<42}: {'FOUND' if path.exists() else 'MISSING'}")

    required = [ROOT, PART11_PATH, PART9_PATH, INIT, VACSV]

    if not all(p.exists() for p in required):
        raise FileNotFoundError(
            "A required Part 46 input is missing."
        )

    print("\n" + "=" * 82)
    print("PART 46 — CHECKPOINT TRANSITION FORENSICS")
    print("=" * 82)
    print(f"PyTorch : {torch.__version__}")
    print(f"Device : {DEVICE}")
    print(f"Validation subset : {VAL_N}")
    print(f"Full volume : {FULL_SHAPE}")
    print(f"Evaluation crop : {CROP_SHAPE}")
    print("Training performed : NO")
    print("Optimizer steps : NO")
    print("Part 15 overwritten : NO")
    print("SPIDER used : NO")
    print("Test set used : NO")
    print(f"Part 15 initialization SHA256 : {sha256_file(INIT)}")

    part11 = load_module(
        PART11_PATH,
        "part11_part46_forensics",
    )

    part9 = load_module(
        PART9_PATH,
        "part9_part46_forensics",
    )

    print("\nPART 46 VALIDATION DATA PRELOAD")
    print("-" * 82)

    cases = load_cases(part11, part9)

    print("\nPART 46 SHAPE SMOKE TEST")
    for i in range(min(3, len(cases))):
        image, mask = cases[i]
        print(
            f"Case {i+1}: "
            f"image={tuple(image.shape)} "
            f"mask={tuple(mask.shape)} "
            f"cropFG={(mask > 0).sum().item()}"
        )

    print("✓ Shape smoke test PASSED.")

    all_rows = []

    all_rows += analyze_condition(
        "part43_random",
        P43R,
        part11,
        cases,
    )

    all_rows += analyze_condition(
        "part43_centered",
        P43C,
        part11,
        cases,
    )

    all_rows += analyze_condition(
        "part44_original",
        P44O,
        part11,
        cases,
    )

    all_rows += analyze_condition(
        "part44_foreground_aware",
        P44A,
        part11,
        cases,
    )

    if not all_rows:
        raise RuntimeError(
            "No Part 43/44 checkpoints were found."
        )

    csv_path = REPORT / "part46_checkpoint_forensics.csv"

    fields = sorted({
        key
        for row in all_rows
        for key in row.keys()
    })

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Transition analysis
    # --------------------------------------------------------

    transitions = []

    by_condition = {}

    for row in all_rows:
        by_condition.setdefault(
            row["condition"],
            []
        ).append(row)

    for condition, rows in by_condition.items():
        rows = sorted(
            rows,
            key=lambda r: (
                0 if r["checkpoint"] == "initialization"
                else int(
                    r["checkpoint"].split("_")[-1]
                )
                if r["checkpoint"].startswith("epoch_")
                and r["checkpoint"].split("_")[-1].isdigit()
                else 99
            )
        )

        prev = None

        for row in rows:
            if prev is not None:
                transitions.append({
                    "condition": condition,
                    "from": prev["checkpoint"],
                    "to": row["checkpoint"],
                    "pred_fg_change": (
                        row["pred_fg"] - prev["pred_fg"]
                    ),
                    "fg_probability_change": (
                        row["fg_probability"]
                        - prev["fg_probability"]
                    ),
                    "fg_logit_gap_target_fg_change": (
                        row["fg_bg_logit_gap_on_target_fg"]
                        - prev["fg_bg_logit_gap_on_target_fg"]
                    ),
                    "gradient_ratio_change": (
                        row["foreground_background_grad_ratio"]
                        - prev["foreground_background_grad_ratio"]
                    ),
                    "parameter_delta_l2": row[
                        "parameter_delta_l2"
                    ],
                })

            prev = row

    transition_path = REPORT / "part46_transitions.json"

    with transition_path.open("w", encoding="utf-8") as f:
        json.dump(transitions, f, indent=2)

    zero_rows = [
        r for r in all_rows
        if r["checkpoint"].startswith("epoch_")
        and r["pred_fg"] == 0
    ]

    first_zero = {}

    for condition, rows in by_condition.items():
        epochs = [
            r for r in rows
            if r["checkpoint"].startswith("epoch_")
            and r["pred_fg"] == 0
        ]

        first_zero[condition] = (
            min(
                epochs,
                key=lambda r: int(
                    r["checkpoint"].split("_")[-1]
                )
            )["checkpoint"]
            if epochs
            else None
        )

    summary = {
        "part": 46,
        "purpose": "checkpoint transition forensics",
        "device": str(DEVICE),
        "validation_subset": VAL_N,
        "crop_shape": CROP_SHAPE,
        "training_performed": False,
        "optimizer_steps": False,
        "part15_overwritten": False,
        "initialization_sha256": sha256_file(INIT),
        "conditions_analyzed": sorted(by_condition.keys()),
        "first_zero_pred_fg_checkpoint": first_zero,
        "total_checkpoint_records": len(all_rows),
        "transition_records": len(transitions),
        "interpretation": (
            "CHECKPOINT_FORENSICS_COMPLETE_REVIEW_TRANSITION_METRICS"
        ),
        "csv": str(csv_path),
        "transitions_json": str(transition_path),
    }

    summary_path = REPORT / "part46_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 82)
    print("PART 46 COMPLETE")
    print("=" * 82)
    print("No training or optimizer updates were performed.")
    print(f"Checkpoint records : {len(all_rows)}")
    print(f"First zero-PredFG by condition : {first_zero}")
    print(f"Forensics CSV : {csv_path}")
    print(f"Transitions JSON : {transition_path}")
    print(f"Summary JSON : {summary_path}")


if __name__ == "__main__":
    main()
