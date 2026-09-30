from pathlib import Path

target = Path(__file__).resolve().parent / "segmentation_rsna_part65_point_to_pseudomask_alignment_forensics.py"

if not target.exists():
    raise FileNotFoundError(f"Part 65 file not found: {target}")

text = target.read_text(encoding="utf-8")

helper = r