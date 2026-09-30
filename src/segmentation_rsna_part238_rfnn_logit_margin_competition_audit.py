"""
PART 2.38 - RFNN LOGIT-MARGIN / CLASS-COMPETITION AUDIT

Purpose
-------
Determine which competing class defeats RFNN at the final output-logit level.

Exact validation cohort:
    Part 2.20B validation cohort

Exact checkpoint:
    Part 2.27 best macro-disease checkpoint

Paired observations:
    LFNN/RFNN pairs from the same study + series + level

For every paired LFNN/RFNN point, calculate final-logit margins:

    RFNN - Background
    RFNN - SCS
    RFNN - LFNN
    RFNN - LSS
    RFNN - RSS

Also determine:

    - RFNN predicted class
    - strongest competing class
    - strongest competing logit
    - RFNN vs strongest competitor margin
    - whether RFNN wins
    - exact class responsible for RFNN failure
    - RFNN probability
    - background probability
    - SCS probability
    - LFNN probability
    - LSS probability
    - RSS probability

This experiment:
    - does NOT train
    - does NOT modify any checkpoint
    - does NOT modify the dashboard
    - does NOT create voxel ground truth
"""


from __future__ import annotations

import json
import sys
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part238_rfnn_logit_margin_competition_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "segmentation"
    / "rsna_part227_multiscale_point_supervised_training"
    / "checkpoints"
    / "part227_best_macro_disease.pth"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_SHAPE = (64, 96, 96)

BACKGROUND = 0
SCS = 1
LFNN = 2
RFNN = 3
LSS = 4
RSS = 5

CLASS_NAMES = {
    0: "Background",
    1: "Spinal Canal Stenosis",
    2: "Left Neural Foraminal Narrowing",
    3: "Right Neural Foraminal Narrowing",
    4: "Left Subarticular Stenosis",
    5: "Right Subarticular Stenosis",
}

# All possible competitors to RFNN.
RFNN_COMPETITORS = [
    BACKGROUND,
    SCS,
    LFNN,
    LSS,
    RSS,
]

# All possible competitors to LFNN.
LFNN_COMPETITORS = [
    BACKGROUND,
    SCS,
    RFNN,
    LSS,
    RSS,
]


# =============================================================================
# IMPORT PART 2.20B
# =============================================================================

def import_part220b():

    candidates = [
        "segmentation_rsna_part220b_geometry_corrected_training",
        "segmentation_rsna_part220b_geometry_corrected_point_supervised_training",
    ]

    errors = []

    for module_name in candidates:

        try:

            module = import_module(
                module_name
            )

            print(
                "[OK] Imported Part 2.20B module:",
                module_name,
            )

            return module

        except Exception as exc:

            errors.append(
                (
                    module_name,
                    repr(exc),
                )
            )

    print(
        "\nCould not import Part 2.20B."
    )

    for module_name, error in errors:

        print(
            f"  {module_name}: {error}"
        )

    raise ImportError(
        "Part 2.20B module could not be imported."
    )


part220b = import_part220b()


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def section(title):

    print(
        "\n"
        + "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


def clean_id(value):

    if pd.isna(value):
        return ""

    try:

        numeric = float(value)

        if numeric.is_integer():

            return str(
                int(numeric)
            )

    except Exception:
        pass

    return str(value).strip()


# =============================================================================
# MODEL
# =============================================================================

def build_model():

    return part220b.build_model()


def load_checkpoint(model):

    section(
        "LOADING PART 2.27 BEST CHECKPOINT"
    )

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    print(
        "Device:",
        DEVICE,
    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    print(
        "Missing keys:",
        len(missing),
    )

    print(
        "Unexpected keys:",
        len(unexpected),
    )

    if missing:

        print(
            "Missing:",
            missing,
        )

    if unexpected:

        print(
            "Unexpected:",
            unexpected,
        )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "[OK] Model loaded."
    )

    print(
        "Parameter count:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    return model


# =============================================================================
# VALIDATION COHORT
# =============================================================================

def recover_validation_points():

    section(
        "LOADING EXACT PART 2.20B VALIDATION COHORT"
    )

    manifest = (
        part220b.load_manifest()
    )

    print(
        "Full manifest rows:",
        len(manifest),
    )

    validation_series = (
        part220b.select_validation_series(
            manifest
        )
    )

    if isinstance(
        validation_series,
        pd.DataFrame,
    ):

        val_df = (
            validation_series.copy()
        )

    else:

        val_df = pd.DataFrame(
            validation_series
        )

    print(
        "Validation series returned:",
        len(val_df),
    )

    def find_column(
        df,
        candidates,
    ):

        lower_map = {
            str(c).lower(): c
            for c in df.columns
        }

        for candidate in candidates:

            if candidate.lower() in lower_map:

                return lower_map[
                    candidate.lower()
                ]

        return None

    study_col = find_column(
        val_df,
        ["study_id"],
    )

    series_col = find_column(
        val_df,
        ["series_id"],
    )

    if (
        study_col is None
        or series_col is None
    ):

        raise KeyError(
            "study_id/series_id missing "
            "from validation output."
        )

    validation_keys = {

        (
            clean_id(
                row[study_col]
            ),

            clean_id(
                row[series_col]
            ),

        )

        for _, row
        in val_df.iterrows()
    }

    manifest_study_col = find_column(
        manifest,
        ["study_id"],
    )

    manifest_series_col = find_column(
        manifest,
        ["series_id"],
    )

    if (
        manifest_study_col is None
        or manifest_series_col is None
    ):

        raise KeyError(
            "study_id/series_id missing "
            "from manifest."
        )

    mask = manifest.apply(

        lambda row:

        (
            clean_id(
                row[
                    manifest_study_col
                ]
            ),

            clean_id(
                row[
                    manifest_series_col
                ]
            ),

        ) in validation_keys,

        axis=1,
    )

    point_rows = (
        manifest.loc[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Recovered validation point rows:",
        len(point_rows),
    )

    print(
        "Validation studies:",
        point_rows[
            "study_id"
        ].nunique(),
    )

    print(
        "Validation series:",
        point_rows[
            "series_id"
        ].nunique(),
    )

    foraminal_mask = (
        point_rows[
            "class_name"
        ]
        .astype(str)
        .str.contains(
            "Neural Foraminal",
            case=False,
            na=False,
        )
    )

    foraminal = (
        point_rows.loc[
            foraminal_mask
        ]
        .copy()
    )

    print(
        "Validation LFNN/RFNN rows:",
        len(foraminal),
    )

    return (
        point_rows,
        foraminal,
    )


# =============================================================================
# PAIR LFNN/RFNN
# =============================================================================

def build_pairs(foraminal):

    pairs = []

    grouped = foraminal.groupby(
        [
            "study_id",
            "series_id",
            "level",
        ],
        dropna=False,
    )

    for (
        study_id,
        series_id,
        level,
    ), group in grouped:

        left = group[
            group[
                "class_name"
            ]
            .astype(str)
            .str.contains(
                "Left Neural",
                case=False,
                na=False,
            )
        ].copy()

        right = group[
            group[
                "class_name"
            ]
            .astype(str)
            .str.contains(
                "Right Neural",
                case=False,
                na=False,
            )
        ].copy()

        if (
            len(left) == 0
            or len(right) == 0
        ):

            continue

        left = (
            left.sort_values(
                [
                    "native_z",
                    "native_y",
                    "native_x",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        right = (
            right.sort_values(
                [
                    "native_z",
                    "native_y",
                    "native_x",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        n = min(
            len(left),
            len(right),
        )

        for index in range(n):

            pairs.append(
                {
                    "pair_id":
                        len(pairs),

                    "study_id":
                        clean_id(
                            study_id
                        ),

                    "series_id":
                        clean_id(
                            series_id
                        ),

                    "level":
                        str(level),

                    "lfnn_row":
                        left.iloc[
                            index
                        ].to_dict(),

                    "rfnn_row":
                        right.iloc[
                            index
                        ].to_dict(),
                }
            )

    return pairs


# =============================================================================
# TRANSFORMED POINT LOOKUP
# =============================================================================

def transformed_point_lookup(
    transformed_points
):

    lookup = {}

    for point in transformed_points:

        key = (
            str(
                point.get(
                    "class_name",
                    "",
                )
            ),
            str(
                point.get(
                    "level",
                    "",
                )
            ),
        )

        lookup.setdefault(
            key,
            [],
        ).append(point)

    return lookup


def get_transformed_point(
    lookup,
    row,
):

    key = (
        str(
            row["class_name"]
        ),
        str(
            row["level"]
        ),
    )

    candidates = lookup.get(
        key,
        [],
    )

    if not candidates:

        raise KeyError(
            f"Transformed point missing: {key}"
        )

    return candidates[0]


# =============================================================================
# POINT
# =============================================================================

def model_point_from_transformed(
    point
):

    return np.array(
        [
            float(
                point["z"]
            ),
            float(
                point["y"]
            ),
            float(
                point["x"]
            ),
        ],
        dtype=np.float32,
    )


# =============================================================================
# LOGIT EXTRACTION
# =============================================================================

def get_output_at_point(
    logits,
    point,
):

    if logits.ndim == 5:

        logits = logits[0]

    if logits.ndim != 4:

        raise ValueError(
            "Expected C x D x H x W logits."
        )

    channels, depth, height, width = (
        logits.shape
    )

    z = int(
        round(
            float(point[0])
        )
    )

    y = int(
        round(
            float(point[1])
        )
    )

    x = int(
        round(
            float(point[2])
        )
    )

    z = max(
        0,
        min(
            depth - 1,
            z,
        ),
    )

    y = max(
        0,
        min(
            height - 1,
            y,
        ),
    )

    x = max(
        0,
        min(
            width - 1,
            x,
        ),
    )

    voxel_logits = (
        logits[
            :,
            z,
            y,
            x,
        ]
    )

    probabilities = torch.softmax(
        voxel_logits,
        dim=0,
    )

    logits_np = (
        voxel_logits
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    probs_np = (
        probabilities
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    predicted_class = int(
        np.argmax(
            probs_np
        )
    )

    return (
        logits_np,
        probs_np,
        predicted_class,
        (
            z,
            y,
            x,
        ),
    )


# =============================================================================
# MODEL INFERENCE
# =============================================================================

@torch.no_grad()
def run_model(
    model,
    image,
):

    if not torch.is_tensor(
        image
    ):

        image = torch.as_tensor(
            image,
            dtype=torch.float32,
        )

    if image.ndim == 3:

        image = image.unsqueeze(0)

    if image.ndim == 4:

        image = image.unsqueeze(0)

    image = image.to(
        DEVICE,
        dtype=torch.float32,
    )

    output = model(
        image
    )

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):

        output = output[0]

    return output


# =============================================================================
# COMPETITION ANALYSIS
# =============================================================================

def analyze_class_competition(
    logits,
    probs,
    target_class,
    competitors,
):

    target_logit = float(
        logits[
            target_class
        ]
    )

    target_probability = float(
        probs[
            target_class
        ]
    )

    competitor_records = []

    for competitor in competitors:

        competitor_logit = float(
            logits[
                competitor
            ]
        )

        margin = (
            target_logit
            - competitor_logit
        )

        competitor_records.append(
            {
                "class_id":
                    competitor,

                "class_name":
                    CLASS_NAMES[
                        competitor
                    ],

                "logit":
                    competitor_logit,

                "margin_target_minus_competitor":
                    margin,
            }
        )

    strongest_competitor = max(
        competitor_records,
        key=lambda item:
        item[
            "logit"
        ],
    )

    strongest_competitor_id = (
        strongest_competitor[
            "class_id"
        ]
    )

    strongest_competitor_logit = (
        strongest_competitor[
            "logit"
        ]
    )

    target_vs_strongest_margin = (
        target_logit
        - strongest_competitor_logit
    )

    target_wins = (
        target_vs_strongest_margin
        > 0
    )

    return {
        "target_logit":
            target_logit,

        "target_probability":
            target_probability,

        "strongest_competitor_id":
            int(
                strongest_competitor_id
            ),

        "strongest_competitor_name":
            CLASS_NAMES[
                strongest_competitor_id
            ],

        "strongest_competitor_logit":
            float(
                strongest_competitor_logit
            ),

        "target_minus_strongest_margin":
            float(
                target_vs_strongest_margin
            ),

        "target_wins":
            bool(
                target_wins
            ),

        "competitor_records":
            competitor_records,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "PART 2.38 - RFNN LOGIT-MARGIN / "
        "CLASS-COMPETITION AUDIT"
    )

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = build_model()

    model = load_checkpoint(
        model
    )

    # -------------------------------------------------------------------------
    # Validation cohort
    # -------------------------------------------------------------------------

    (
        point_rows,
        foraminal,
    ) = recover_validation_points()

    pairs = build_pairs(
        foraminal
    )

    section(
        "PAIRING SUMMARY"
    )

    candidate_series = {
        (
            pair["study_id"],
            pair["series_id"],
        )
        for pair in pairs
    }

    print(
        "Candidate paired series:",
        len(candidate_series),
    )

    print(
        "Paired LFNN/RFNN observations:",
        len(pairs),
    )

    # -------------------------------------------------------------------------
    # Records
    # -------------------------------------------------------------------------

    rfnn_records = []

    lfnn_records = []

    successful_cases = set()

    unique_cases = sorted(
        candidate_series
    )

    # -------------------------------------------------------------------------
    # Cases
    # -------------------------------------------------------------------------

    for case_index, (
        study_id,
        series_id,
    ) in enumerate(
        unique_cases,
        start=1,
    ):

        print(
            f"\n[{case_index}/{len(unique_cases)}] "
            f"Study={study_id} "
            f"Series={series_id}"
        )

        case_rows = point_rows[
            (
                point_rows[
                    "study_id"
                ].map(clean_id)
                == study_id
            )
            &
            (
                point_rows[
                    "series_id"
                ].map(clean_id)
                == series_id
            )
        ].copy()

        try:

            image, transformed_points, geometry = (
                part220b.load_case(
                    study_id,
                    series_id,
                    case_rows,
                )
            )

            print(
                "  Image shape:",
                tuple(
                    image.shape
                ),
            )

            print(
                "  Original point rows:",
                len(case_rows),
            )

            print(
                "  Transformed points:",
                len(
                    transformed_points
                ),
            )

            lookup = (
                transformed_point_lookup(
                    transformed_points
                )
            )

            logits = run_model(
                model,
                image,
            )

            print(
                "  Output shape:",
                tuple(
                    logits.shape
                ),
            )

            case_pairs = [
                pair
                for pair in pairs
                if (
                    pair["study_id"]
                    == study_id
                    and pair["series_id"]
                    == series_id
                )
            ]

            # -------------------------------------------------------------
            # Paired LFNN/RFNN observations
            # -------------------------------------------------------------

            for pair in case_pairs:

                lrow = pair[
                    "lfnn_row"
                ]

                rrow = pair[
                    "rfnn_row"
                ]

                lpoint_dict = (
                    get_transformed_point(
                        lookup,
                        lrow,
                    )
                )

                rpoint_dict = (
                    get_transformed_point(
                        lookup,
                        rrow,
                    )
                )

                lpoint = (
                    model_point_from_transformed(
                        lpoint_dict
                    )
                )

                rpoint = (
                    model_point_from_transformed(
                        rpoint_dict
                    )
                )

                # ---------------------------------------------------------
                # LFNN
                # ---------------------------------------------------------

                (
                    l_logits,
                    l_probs,
                    l_pred,
                    l_voxel,
                ) = get_output_at_point(
                    logits,
                    lpoint,
                )

                l_analysis = (
                    analyze_class_competition(
                        l_logits,
                        l_probs,
                        LFNN,
                        LFNN_COMPETITORS,
                    )
                )

                l_record = {
                    "pair_id":
                        pair[
                            "pair_id"
                        ],

                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "level":
                        str(
                            pair[
                                "level"
                            ]
                        ),

                    "target_class":
                        LFNN,

                    "target_class_name":
                        CLASS_NAMES[
                            LFNN
                        ],

                    "sampled_z":
                        l_voxel[0],

                    "sampled_y":
                        l_voxel[1],

                    "sampled_x":
                        l_voxel[2],

                    "predicted_class":
                        l_pred,

                    "predicted_class_name":
                        CLASS_NAMES.get(
                            l_pred,
                            f"Class_{l_pred}",
                        ),

                    "target_logit":
                        l_analysis[
                            "target_logit"
                        ],

                    "target_probability":
                        l_analysis[
                            "target_probability"
                        ],

                    "strongest_competitor_id":
                        l_analysis[
                            "strongest_competitor_id"
                        ],

                    "strongest_competitor_name":
                        l_analysis[
                            "strongest_competitor_name"
                        ],

                    "strongest_competitor_logit":
                        l_analysis[
                            "strongest_competitor_logit"
                        ],

                    "target_minus_strongest_margin":
                        l_analysis[
                            "target_minus_strongest_margin"
                        ],

                    "target_wins":
                        l_analysis[
                            "target_wins"
                        ],

                    "background_logit":
                        float(
                            l_logits[
                                BACKGROUND
                            ]
                        ),

                    "scs_logit":
                        float(
                            l_logits[
                                SCS
                            ]
                        ),

                    "lfnn_logit":
                        float(
                            l_logits[
                                LFNN
                            ]
                        ),

                    "rfnn_logit":
                        float(
                            l_logits[
                                RFNN
                            ]
                        ),

                    "lss_logit":
                        float(
                            l_logits[
                                LSS
                            ]
                        ),

                    "rss_logit":
                        float(
                            l_logits[
                                RSS
                            ]
                        ),

                    "background_probability":
                        float(
                            l_probs[
                                BACKGROUND
                            ]
                        ),

                    "scs_probability":
                        float(
                            l_probs[
                                SCS
                            ]
                        ),

                    "lfnn_probability":
                        float(
                            l_probs[
                                LFNN
                            ]
                        ),

                    "rfnn_probability":
                        float(
                            l_probs[
                                RFNN
                            ]
                        ),

                    "lss_probability":
                        float(
                            l_probs[
                                LSS
                            ]
                        ),

                    "rss_probability":
                        float(
                            l_probs[
                                RSS
                            ]
                        ),
                }

                l_record[
                    "lfnn_minus_background_logit"
                ] = (
                    l_logits[
                        LFNN
                    ]
                    - l_logits[
                        BACKGROUND
                    ]
                )

                l_record[
                    "lfnn_minus_scs_logit"
                ] = (
                    l_logits[
                        LFNN
                    ]
                    - l_logits[
                        SCS
                    ]
                )

                l_record[
                    "lfnn_minus_rfnn_logit"
                ] = (
                    l_logits[
                        LFNN
                    ]
                    - l_logits[
                        RFNN
                    ]
                )

                l_record[
                    "lfnn_minus_lss_logit"
                ] = (
                    l_logits[
                        LFNN
                    ]
                    - l_logits[
                        LSS
                    ]
                )

                l_record[
                    "lfnn_minus_rss_logit"
                ] = (
                    l_logits[
                        LFNN
                    ]
                    - l_logits[
                        RSS
                    ]
                )

                lfnn_records.append(
                    l_record
                )

                # ---------------------------------------------------------
                # RFNN
                # ---------------------------------------------------------

                (
                    r_logits,
                    r_probs,
                    r_pred,
                    r_voxel,
                ) = get_output_at_point(
                    logits,
                    rpoint,
                )

                r_analysis = (
                    analyze_class_competition(
                        r_logits,
                        r_probs,
                        RFNN,
                        RFNN_COMPETITORS,
                    )
                )

                r_record = {
                    "pair_id":
                        pair[
                            "pair_id"
                        ],

                    "study_id":
                        study_id,

                    "series_id":
                        series_id,

                    "level":
                        str(
                            pair[
                                "level"
                            ]
                        ),

                    "target_class":
                        RFNN,

                    "target_class_name":
                        CLASS_NAMES[
                            RFNN
                        ],

                    "sampled_z":
                        r_voxel[0],

                    "sampled_y":
                        r_voxel[1],

                    "sampled_x":
                        r_voxel[2],

                    "predicted_class":
                        r_pred,

                    "predicted_class_name":
                        CLASS_NAMES.get(
                            r_pred,
                            f"Class_{r_pred}",
                        ),

                    "target_logit":
                        r_analysis[
                            "target_logit"
                        ],

                    "target_probability":
                        r_analysis[
                            "target_probability"
                        ],

                    "strongest_competitor_id":
                        r_analysis[
                            "strongest_competitor_id"
                        ],

                    "strongest_competitor_name":
                        r_analysis[
                            "strongest_competitor_name"
                        ],

                    "strongest_competitor_logit":
                        r_analysis[
                            "strongest_competitor_logit"
                        ],

                    "target_minus_strongest_margin":
                        r_analysis[
                            "target_minus_strongest_margin"
                        ],

                    "target_wins":
                        r_analysis[
                            "target_wins"
                        ],

                    "background_logit":
                        float(
                            r_logits[
                                BACKGROUND
                            ]
                        ),

                    "scs_logit":
                        float(
                            r_logits[
                                SCS
                            ]
                        ),

                    "lfnn_logit":
                        float(
                            r_logits[
                                LFNN
                            ]
                        ),

                    "rfnn_logit":
                        float(
                            r_logits[
                                RFNN
                            ]
                        ),

                    "lss_logit":
                        float(
                            r_logits[
                                LSS
                            ]
                        ),

                    "rss_logit":
                        float(
                            r_logits[
                                RSS
                            ]
                        ),

                    "background_probability":
                        float(
                            r_probs[
                                BACKGROUND
                            ]
                        ),

                    "scs_probability":
                        float(
                            r_probs[
                                SCS
                            ]
                        ),

                    "lfnn_probability":
                        float(
                            r_probs[
                                LFNN
                            ]
                        ),

                    "rfnn_probability":
                        float(
                            r_probs[
                                RFNN
                            ]
                        ),

                    "lss_probability":
                        float(
                            r_probs[
                                LSS
                            ]
                        ),

                    "rss_probability":
                        float(
                            r_probs[
                                RSS
                            ]
                        ),
                }

                r_record[
                    "rfnn_minus_background_logit"
                ] = (
                    r_logits[
                        RFNN
                    ]
                    - r_logits[
                        BACKGROUND
                    ]
                )

                r_record[
                    "rfnn_minus_scs_logit"
                ] = (
                    r_logits[
                        RFNN
                    ]
                    - r_logits[
                        SCS
                    ]
                )

                r_record[
                    "rfnn_minus_lfnn_logit"
                ] = (
                    r_logits[
                        RFNN
                    ]
                    - r_logits[
                        LFNN
                    ]
                )

                r_record[
                    "rfnn_minus_lss_logit"
                ] = (
                    r_logits[
                        RFNN
                    ]
                    - r_logits[
                        LSS
                    ]
                )

                r_record[
                    "rfnn_minus_rss_logit"
                ] = (
                    r_logits[
                        RFNN
                    ]
                    - r_logits[
                        RSS
                    ]
                )

                rfnn_records.append(
                    r_record
                )

            successful_cases.add(
                (
                    study_id,
                    series_id,
                )
            )

        except Exception as exc:

            print(
                "  [ERROR]",
                repr(exc),
            )

    # =============================================================================
    # DATAFRAMES
    # =============================================================================

    rfnn_df = pd.DataFrame(
        rfnn_records
    )

    lfnn_df = pd.DataFrame(
        lfnn_records
    )

    if rfnn_df.empty:

        raise RuntimeError(
            "No RFNN records generated."
        )

    if lfnn_df.empty:

        raise RuntimeError(
            "No LFNN records generated."
        )

    # =============================================================================
    # RFNN COMPETITION SUMMARY
    # =============================================================================

    section(
        "RFNN CLASS-COMPETITION SUMMARY"
    )

    rfnn_competition = (
        rfnn_df[
            "strongest_competitor_name"
        ]
        .value_counts()
        .rename_axis(
            "strongest_competitor"
        )
        .reset_index(
            name="count"
        )
    )

    rfnn_competition[
        "fraction"
    ] = (
        rfnn_competition[
            "count"
        ]
        / len(rfnn_df)
    )

    print(
        rfnn_competition.to_string(
            index=False
        )
    )

    # =============================================================================
    # RFNN WIN/LOSS
    # =============================================================================

    rfnn_wins = int(
        rfnn_df[
            "target_wins"
        ].sum()
    )

    rfnn_losses = (
        len(rfnn_df)
        - rfnn_wins
    )

    print(
        "\nRFNN target wins:",
        rfnn_wins,
        "/",
        len(rfnn_df),
    )

    print(
        "RFNN target loses:",
        rfnn_losses,
        "/",
        len(rfnn_df),
    )

    # =============================================================================
    # RFNN MARGINS
    # =============================================================================

    margin_columns = [
        "rfnn_minus_background_logit",
        "rfnn_minus_scs_logit",
        "rfnn_minus_lfnn_logit",
        "rfnn_minus_lss_logit",
        "rfnn_minus_rss_logit",
    ]

    margin_summary_records = []

    for column in margin_columns:

        values = rfnn_df[
            column
        ].to_numpy(
            dtype=float
        )

        margin_summary_records.append(
            {
                "margin":
                    column,

                "mean":
                    float(
                        np.mean(values)
                    ),

                "median":
                    float(
                        np.median(values)
                    ),

                "std":
                    float(
                        np.std(values)
                    ),

                "minimum":
                    float(
                        np.min(values)
                    ),

                "maximum":
                    float(
                        np.max(values)
                    ),

                "positive_count":
                    int(
                        np.sum(
                            values > 0
                        )
                    ),

                "negative_count":
                    int(
                        np.sum(
                            values < 0
                        )
                    ),

                "zero_count":
                    int(
                        np.sum(
                            values == 0
                        )
                    ),
            }
        )

    margin_summary_df = pd.DataFrame(
        margin_summary_records
    )

    print(
        "\nRFNN logit-margin summary:"
    )

    print(
        margin_summary_df.to_string(
            index=False
        )
    )

    # =============================================================================
    # PAIRED LFNN VS RFNN SUMMARY
    # =============================================================================

    paired = (
        lfnn_df[
            [
                "pair_id",
                "study_id",
                "series_id",
                "level",
                "target_probability",
                "target_logit",
                "strongest_competitor_name",
                "target_minus_strongest_margin",
                "predicted_class",
                "predicted_class_name",
            ]
        ]
        .rename(
            columns={
                "target_probability":
                    "lfnn_target_probability",

                "target_logit":
                    "lfnn_target_logit",

                "strongest_competitor_name":
                    "lfnn_strongest_competitor",

                "target_minus_strongest_margin":
                    "lfnn_target_margin",

                "predicted_class":
                    "lfnn_predicted_class",

                "predicted_class_name":
                    "lfnn_predicted_class_name",
            }
        )
        .merge(
            rfnn_df[
                [
                    "pair_id",
                    "target_probability",
                    "target_logit",
                    "strongest_competitor_name",
                    "target_minus_strongest_margin",
                    "predicted_class",
                    "predicted_class_name",
                ]
            ].rename(
                columns={
                    "target_probability":
                        "rfnn_target_probability",

                    "target_logit":
                        "rfnn_target_logit",

                    "strongest_competitor_name":
                        "rfnn_strongest_competitor",

                    "target_minus_strongest_margin":
                        "rfnn_target_margin",

                    "predicted_class":
                        "rfnn_predicted_class",

                    "predicted_class_name":
                        "rfnn_predicted_class_name",
                }
            ),
            on="pair_id",
            how="inner",
        )
    )

    paired[
        "rfnn_minus_lfnn_target_probability"
    ] = (
        paired[
            "rfnn_target_probability"
        ]
        - paired[
            "lfnn_target_probability"
        ]
    )

    paired[
        "rfnn_minus_lfnn_target_logit"
    ] = (
        paired[
            "rfnn_target_logit"
        ]
        - paired[
            "lfnn_target_logit"
        ]
    )

    # =============================================================================
    # PER-DISEASE / COMPETITOR SUMMARY
    # =============================================================================

    disease_summary = pd.DataFrame(
        [
            {
                "target":
                    "RFNN",

                "points":
                    len(rfnn_df),

                "accuracy":
                    float(
                        rfnn_df[
                            "target_wins"
                        ].mean()
                    ),

                "mean_target_probability":
                    float(
                        rfnn_df[
                            "target_probability"
                        ].mean()
                    ),

                "median_target_probability":
                    float(
                        rfnn_df[
                            "target_probability"
                        ].median()
                    ),

                "mean_target_minus_strongest_margin":
                    float(
                        rfnn_df[
                            "target_minus_strongest_margin"
                        ].mean()
                    ),

                "median_target_minus_strongest_margin":
                    float(
                        rfnn_df[
                            "target_minus_strongest_margin"
                        ].median()
                    ),
            },

            {
                "target":
                    "LFNN",

                "points":
                    len(lfnn_df),

                "accuracy":
                    float(
                        lfnn_df[
                            "target_wins"
                        ].mean()
                    ),

                "mean_target_probability":
                    float(
                        lfnn_df[
                            "target_probability"
                        ].mean()
                    ),

                "median_target_probability":
                    float(
                        lfnn_df[
                            "target_probability"
                        ].median()
                    ),

                "mean_target_minus_strongest_margin":
                    float(
                        lfnn_df[
                            "target_minus_strongest_margin"
                        ].mean()
                    ),

                "median_target_minus_strongest_margin":
                    float(
                        lfnn_df[
                            "target_minus_strongest_margin"
                        ].median()
                    ),
            },
        ]
    )

    # =============================================================================
    # SAVE FILES
    # =============================================================================

    rfnn_path = (
        OUTPUT_DIR
        / "part238_rfnn_point_competition_records.csv"
    )

    lfnn_path = (
        OUTPUT_DIR
        / "part238_lfnn_point_competition_records.csv"
    )

    competition_path = (
        OUTPUT_DIR
        / "part238_rfnn_competition_summary.csv"
    )

    margin_path = (
        OUTPUT_DIR
        / "part238_rfnn_logit_margin_summary.csv"
    )

    paired_path = (
        OUTPUT_DIR
        / "part238_paired_lfnn_rfnn_margin_comparison.csv"
    )

    disease_path = (
        OUTPUT_DIR
        / "part238_lfnn_rfnn_disease_summary.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "part238_logit_margin_audit_summary.json"
    )

    rfnn_df.to_csv(
        rfnn_path,
        index=False,
    )

    lfnn_df.to_csv(
        lfnn_path,
        index=False,
    )

    rfnn_competition.to_csv(
        competition_path,
        index=False,
    )

    margin_summary_df.to_csv(
        margin_path,
        index=False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    disease_summary.to_csv(
        disease_path,
        index=False,
    )

    # =============================================================================
    # JSON
    # =============================================================================

    json_summary = {

        "part":
            "2.38",

        "title":
            "RFNN Logit-Margin / Class-Competition Audit",

        "purpose":
            (
                "Determine which competing segmentation class "
                "defeats RFNN at the final output-logit level."
            ),

        "checkpoint":
            str(
                CHECKPOINT
            ),

        "device":
            str(
                DEVICE
            ),

        "model_shape":
            list(
                MODEL_SHAPE
            ),

        "successful_cases":
            len(
                successful_cases
            ),

        "paired_observations":
            len(
                pairs
            ),

        "rfnn_points":
            len(
                rfnn_df
            ),

        "lfnn_points":
            len(
                lfnn_df
            ),

        "rfnn_wins":
            rfnn_wins,

        "rfnn_losses":
            rfnn_losses,

        "rfnn_competition":
            rfnn_competition.to_dict(
                orient="records"
            ),

        "files":
            {
                "rfnn_records":
                    str(
                        rfnn_path
                    ),

                "lfnn_records":
                    str(
                        lfnn_path
                    ),

                "competition_summary":
                    str(
                        competition_path
                    ),

                "margin_summary":
                    str(
                        margin_path
                    ),

                "paired_comparison":
                    str(
                        paired_path
                    ),

                "disease_summary":
                    str(
                        disease_path
                    ),
            },

        "training_performed":
            False,

        "checkpoint_modified":
            False,

        "dashboard_modified":
            False,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
        )

    # =============================================================================
    # FINAL REPORT
    # =============================================================================

    section(
        "PART 2.38 COMPLETE"
    )

    print(
        "Successful cases:",
        len(
            successful_cases
        ),
    )

    print(
        "Paired observations:",
        len(
            pairs
        ),
    )

    print(
        "RFNN records:",
        len(
            rfnn_df
        ),
    )

    print(
        "LFNN records:",
        len(
            lfnn_df
        ),
    )

    print(
        "RFNN wins:",
        rfnn_wins,
    )

    print(
        "RFNN losses:",
        rfnn_losses,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {rfnn_path}"
    )

    print(
        f"  {lfnn_path}"
    )

    print(
        f"  {competition_path}"
    )

    print(
        f"  {margin_path}"
    )

    print(
        f"  {paired_path}"
    )

    print(
        f"  {disease_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This experiment is analysis-only."
    )

    print(
        "No training/checkpoint/dashboard changes were made."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()