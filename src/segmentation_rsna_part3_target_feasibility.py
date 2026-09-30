from pathlib import Path
import json
import math
import sys
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = PROJECT_ROOT / "dataset" / "rsna-2024-lumbar-spine-degenerative-classification"
TRAIN_IMAGES = RSNA_ROOT / "train_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "segmentation" / "rsna_part3_target_feasibility"

TRAIN_CSV = RSNA_ROOT / "train.csv"
COORD_CSV = RSNA_ROOT / "train_label_coordinates.csv"
SERIES_CSV = RSNA_ROOT / "train_series_descriptions.csv"

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
SERIES_TYPES = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]
CONDITIONS = [
    "Spinal Canal Stenosis",
    "Left Neural Foraminal Narrowing",
    "Right Neural Foraminal Narrowing",
    "Left Subarticular Stenosis",
    "Right Subarticular Stenosis",
]

def banner(s):
    print("=" * 78)
    print(s)
    print("=" * 78)

def norm_level(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper().replace(" ", "").replace("_", "/").replace("-", "/")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("PHASE 4 - PART 3")
    print("RSNA POINT-ANNOTATION SEGMENTATION TARGET FEASIBILITY ANALYSIS")
    banner("")
    print("PROJECT ROOT\n", PROJECT_ROOT)
    print("\nRSNA DATASET\n", RSNA_ROOT)
    print("\nOUTPUT DIRECTORY\n", OUTPUT_DIR)

    banner("DATASET PATH VALIDATION")
    paths = {
        "RSNA root": RSNA_ROOT,
        "train_images": TRAIN_IMAGES,
        "train.csv": TRAIN_CSV,
        "train_label_coordinates.csv": COORD_CSV,
        "train_series_descriptions.csv": SERIES_CSV,
    }
    for name, p in paths.items():
        print(f"{name:<32}: {'FOUND' if p.exists() else 'MISSING'}")
    if not all(p.exists() for p in paths.values()):
        raise FileNotFoundError("Required RSNA dataset files/directories are missing.")

    banner("LOADING RSNA METADATA")
    train = pd.read_csv(TRAIN_CSV)
    coords = pd.read_csv(COORD_CSV)
    series = pd.read_csv(SERIES_CSV)
    print(f"train.csv rows                : {len(train)}")
    print(f"train_label_coordinates rows  : {len(coords)}")
    print(f"train_series_descriptions rows: {len(series)}")

    required_c = {"study_id","series_id","instance_number","condition","level","x","y"}
    required_s = {"study_id","series_id","series_description"}
    if not required_c.issubset(coords.columns):
        raise ValueError(f"Missing coordinate columns: {sorted(required_c-set(coords.columns))}")
    if not required_s.issubset(series.columns):
        raise ValueError(f"Missing series columns: {sorted(required_s-set(series.columns))}")

    coords["study_id"] = coords["study_id"].astype(str)
    coords["series_id"] = coords["series_id"].astype(str)
    coords["condition"] = coords["condition"].astype(str).str.strip()
    coords["level"] = coords["level"].map(norm_level)
    coords["x"] = pd.to_numeric(coords["x"], errors="coerce")
    coords["y"] = pd.to_numeric(coords["y"], errors="coerce")
    coords["instance_number"] = pd.to_numeric(coords["instance_number"], errors="coerce")

    series["study_id"] = series["study_id"].astype(str)
    series["series_id"] = series["series_id"].astype(str)
    series["series_description"] = series["series_description"].astype(str).str.strip()

    df = coords.merge(series, on=["study_id","series_id"], how="left", validate="many_to_one")
    df["series_description"] = df["series_description"].fillna("UNKNOWN")

    banner("ANNOTATION INTEGRITY")
    missing_x = int(df.x.isna().sum())
    missing_y = int(df.y.isna().sum())
    missing_instance = int(df.instance_number.isna().sum())
    missing_series = int((df.series_description == "UNKNOWN").sum())
    duplicates = int(df.duplicated(["study_id","series_id","instance_number","condition","level","x","y"]).sum())
    print(f"Total annotation rows : {len(df)}")
    print(f"Missing X              : {missing_x}")
    print(f"Missing Y              : {missing_y}")
    print(f"Missing instance       : {missing_instance}")
    print(f"Missing series mapping : {missing_series}")
    print(f"Exact duplicate rows   : {duplicates}")

    banner("DICOM COORDINATE BOUNDS AUDIT")
    try:
        import pydicom
    except ImportError:
        pydicom = None
        print("pydicom not available: DICOM geometry audit skipped.")
    geometry = {}
    out_of_bounds = 0
    geometry_missing = 0

    for study, sid in df[["study_id","series_id"]].drop_duplicates().itertuples(index=False):
        d = TRAIN_IMAGES / study / sid
        files = list(d.glob("*.dcm")) if d.exists() else []
        if not files or pydicom is None:
            geometry[(study,sid)] = None
            continue
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            geometry[(study,sid)] = (int(ds.Rows), int(ds.Columns), len(files))
        except Exception:
            geometry[(study,sid)] = None

    valid_geom = 0
    for r in df.itertuples():
        g = geometry.get((r.study_id, r.series_id))
        if g is None:
            geometry_missing += 1
            continue
        valid_geom += 1
        rows, cols, _ = g
        if pd.isna(r.x) or pd.isna(r.y) or not (0 <= r.x < cols and 0 <= r.y < rows):
            out_of_bounds += 1

    print(f"Unique annotated series      : {df[['study_id','series_id']].drop_duplicates().shape[0]}")
    print(f"Annotations with geometry    : {valid_geom}")
    print(f"Annotations without geometry : {geometry_missing}")
    print(f"Coordinate out-of-bounds     : {out_of_bounds}")

    banner("ANNOTATION COVERAGE")
    condition_counts = df.condition.value_counts()
    level_counts = df.level.value_counts()
    series_counts = df.series_description.value_counts()
    print("\nCONDITIONS")
    print(condition_counts.to_string())
    print("\nLEVELS")
    print(level_counts.to_string())
    print("\nSERIES TYPES")
    print(series_counts.to_string())

    coverage = pd.crosstab(df.condition, df.level).reindex(
        index=CONDITIONS, columns=LEVELS, fill_value=0
    )
    print("\nCONDITION x LEVEL")
    print(coverage.to_string())

    banner("POINT GEOMETRY / LEVEL ORDER AUDIT")
    rows = []
    for (study, sid, condition), g in df.groupby(["study_id","series_id","condition"]):
        means = g.groupby("level")["y"].mean().to_dict()
        vals = [means[x] for x in LEVELS if x in means]
        monotonic = None
        if len(vals) >= 3:
            monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1)) or \
                        all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        rows.append({
            "study_id": study, "series_id": sid, "condition": condition,
            "levels_observed": len(vals), "y_order_monotonic": monotonic,
            "y_range": (max(vals)-min(vals)) if len(vals) >= 2 else np.nan
        })
    consistency = pd.DataFrame(rows)
    rate = float(consistency.y_order_monotonic.dropna().mean()) if not consistency.empty and consistency.y_order_monotonic.notna().any() else np.nan
    print("Monotonic lumbar-level y-order rate:", "N/A" if pd.isna(rate) else f"{rate:.4f}")

    banner("FEASIBILITY DECISION")
    checks = {
        "RSNA metadata available": True,
        "Point coordinates available": len(df) > 0,
        "All expected conditions represented": all(int(condition_counts.get(c,0)) > 0 for c in CONDITIONS),
        "All expected lumbar levels represented": all(int(level_counts.get(l,0)) > 0 for l in LEVELS),
        "Expected MRI series represented": all(int(series_counts.get(s,0)) > 0 for s in SERIES_TYPES),
        "No missing X/Y coordinates": missing_x == 0 and missing_y == 0,
        "No coordinate bounds violations": out_of_bounds == 0,
    }
    for k,v in checks.items():
        print(("PASS" if v else "FAIL") + " - " + k)

    print("\nSCIENTIFIC INTERPRETATION")
    print("RSNA point annotations support anatomical localization and weak/semi-supervised target construction.")
    print("They are NOT equivalent to manually drawn pixel-wise segmentation masks.")
    print("Therefore this phase creates NO segmentation masks and performs NO model training.")
    print("A later phase must define, validate, and document a reproducible point-to-region pseudo-target strategy.")

    banner("SAVING ANALYSIS TABLES")
    df["target_category"] = df.condition.map({
        "Spinal Canal Stenosis":"spinal_canal",
        "Left Neural Foraminal Narrowing":"left_neural_foramen",
        "Right Neural Foraminal Narrowing":"right_neural_foramen",
        "Left Subarticular Stenosis":"left_subarticular",
        "Right Subarticular Stenosis":"right_subarticular",
    })
    df.to_csv(OUTPUT_DIR/"rsna_part3_annotation_statistics.csv", index=False)
    coverage.reset_index().to_csv(OUTPUT_DIR/"rsna_part3_condition_level_coverage.csv", index=False)
    consistency.to_csv(OUTPUT_DIR/"rsna_part3_point_geometry_consistency.csv", index=False)

    study_rows = []
    for study, g in df.groupby("study_id"):
        st = set(g.series_description)
        co = set(g.condition)
        le = set(g.level)
        study_rows.append({
            "study_id":study, "annotation_count":len(g),
            "series_count":g.series_id.nunique(), "series_types_present":len(st),
            "conditions_present":len(co), "levels_present":len(le),
            "has_sagittal_t1":"Sagittal T1" in st,
            "has_sagittal_t2_stir":"Sagittal T2/STIR" in st,
            "has_axial_t2":"Axial T2" in st,
            "has_spinal_canal":"Spinal Canal Stenosis" in co,
        })
    pd.DataFrame(study_rows).to_csv(OUTPUT_DIR/"rsna_part3_study_statistics.csv", index=False)

    series_rows = []
    for (study,sid,desc), g in df.groupby(["study_id","series_id","series_description"]):
        geo = geometry.get((study,sid))
        series_rows.append({
            "study_id":study, "series_id":sid, "series_description":desc,
            "annotation_count":len(g), "unique_conditions":g.condition.nunique(),
            "unique_levels":g.level.nunique(), "x_mean":g.x.mean(),
            "x_std":g.x.std(), "y_mean":g.y.mean(), "y_std":g.y.std(),
            "dicom_geometry_available":geo is not None,
            "dicom_image_count":geo[2] if geo else 0,
            "rows":geo[0] if geo else np.nan, "columns":geo[1] if geo else np.nan,
        })
    pd.DataFrame(series_rows).to_csv(OUTPUT_DIR/"rsna_part3_series_statistics.csv", index=False)

    summary = {
        "phase":"Phase 4 - Part 3",
        "title":"RSNA Point-Annotation Segmentation Target Feasibility Analysis",
        "dataset":"RSNA 2024 Lumbar Spine Degenerative Classification",
        "spider_used":False, "training_performed":False,
        "masks_created":False, "model_weights_modified":False,
        "train_rows":len(train), "annotation_rows":len(df), "series_rows":len(series),
        "annotated_studies":int(df.study_id.nunique()),
        "annotated_series":int(df[["study_id","series_id"]].drop_duplicates().shape[0]),
        "missing_x":missing_x, "missing_y":missing_y,
        "missing_instance_number":missing_instance,
        "missing_series_mapping":missing_series,
        "exact_duplicate_rows":duplicates,
        "coordinate_out_of_bounds":out_of_bounds,
        "condition_counts":{k:int(condition_counts.get(k,0)) for k in CONDITIONS},
        "level_counts":{k:int(level_counts.get(k,0)) for k in LEVELS},
        "series_counts":{k:int(series_counts.get(k,0)) for k in SERIES_TYPES},
        "monotonic_level_y_order_rate":None if pd.isna(rate) else rate,
        "checks":checks,
        "conclusion":"RSNA-only segmentation is feasible only with explicitly validated point-derived/weakly supervised targets; point annotations must not be called manual ground truth.",
        "next_phase":"Design and validate the RSNA point-to-region pseudo-target construction method before Swin-UNETR training."
    }
    with open(OUTPUT_DIR/"rsna_part3_target_feasibility_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    report = [
        "PHASE 4 - PART 3",
        "RSNA POINT-ANNOTATION SEGMENTATION TARGET FEASIBILITY ANALYSIS",
        "",
        f"RSNA root: {RSNA_ROOT}",
        f"Train rows: {len(train)}",
        f"Annotation rows: {len(df)}",
        f"Series rows: {len(series)}",
        f"Annotated studies: {df.study_id.nunique()}",
        f"Annotated series: {df[['study_id','series_id']].drop_duplicates().shape[0]}",
        "",
        "ANNOTATION INTEGRITY",
        f"Missing X: {missing_x}",
        f"Missing Y: {missing_y}",
        f"Missing instance: {missing_instance}",
        f"Missing series mapping: {missing_series}",
        f"Exact duplicates: {duplicates}",
        f"Coordinate out-of-bounds: {out_of_bounds}",
        "",
        "SCIENTIFIC CONCLUSION",
        "RSNA can be the sole dataset for the segmentation study.",
        "The coordinate annotations are localization/weak labels, not manual masks.",
        "Any point-derived region must be called a pseudo-label/generated target.",
        "Target generation must be validated before Swin-UNETR training.",
        "",
        "SPIDER used: NO",
        "Training performed: NO",
        "Masks created: NO",
        "Model weights modified: NO",
    ]
    (OUTPUT_DIR/"phase4_part3_target_feasibility_report.txt").write_text("\n".join(report),encoding="utf-8")

    banner("PART 3 COMPLETE")
    print(f"RSNA annotation rows         : {len(df)}")
    print(f"Annotated studies            : {df.study_id.nunique()}")
    print(f"Annotated series             : {df[['study_id','series_id']].drop_duplicates().shape[0]}")
    print(f"Coordinate bounds violations : {out_of_bounds}")
    print("RSNA-only segmentation: FEASIBLE WITH VALIDATED PSEUDO-TARGETS")
    print("SPIDER used: NO")
    print("Training performed: NO")
    print("Masks created: NO")
    print("Model weights modified: NO")
    print("\nOUTPUT DIRECTORY")
    print(OUTPUT_DIR)
    banner("PHASE 4 - PART 3 COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)
