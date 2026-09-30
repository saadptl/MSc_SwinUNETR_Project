from pathlib import Path

path = Path(
    r".\src\segmentation_rsna_part25_dice_api_metric_implementation_audit.py"
)

text = path.read_text(encoding="utf-8")

old = '''def call_part11_dice(
    part11,
    logits,
    mask,
    num_classes,
):'''

new = '''def call_part11_dice(
    part11,
    logits,
    mask,
):'''

if old in text:
    text = text.replace(old, new, 1)

# Remove the old three-argument call patterns.
text = text.replace(
    "part11.dice_from_prediction(logits, mask, num_classes)",
    "part11.dice_from_prediction(logits, mask)",
)

text = text.replace(
    "part11.dice_from_prediction(" + "\n"
    + "            logits," + "\n"
    + "            mask," + "\n"
    + "            num_classes," + "\n"
    + "        )",
    "part11.dice_from_prediction(" + "\n"
    + "            logits," + "\n"
    + "            mask," + "\n"
    + "        )",
)

# Replace the entire old function by locating its boundaries.
start = text.find("def call_part11_dice(")

if start == -1:
    raise RuntimeError("call_part11_dice() was not found.")

# Find the next top-level function.
next_def = text.find("\ndef ", start + 1)

if next_def == -1:
    raise RuntimeError("Could not find end of call_part11_dice().")

replacement = '''def call_part11_dice(part11, logits, mask):
    """Call the validated Part 11 Dice API exactly as implemented."""
    attempts = []

    try:
        value = part11.dice_from_prediction(
            logits,
            mask,
        )

        attempts.append(
            {
                "attempt": "logits, mask",
                "success": True,
                "type": str(type(value)),
            }
        )

        return value, attempts

    except Exception as e:
        attempts.append(
            {
                "attempt": "logits, mask",
                "success": False,
                "error": repr(e),
            }
        )

    raise RuntimeError(
        "Could not call Part 11 dice_from_prediction().\\n"
        + json.dumps(attempts, indent=2)
    )


'''

text = (
    text[:start]
    + replacement
    + text[next_def + 1:]
)

path.write_text(text, encoding="utf-8")

print("=" * 70)
print("PART 25 PATCH SUCCESSFUL")
print("=" * 70)
print("File:", path)
print()
print("Correct Part 11 API:")
print("    dice_from_prediction(logits, target)")
print()
print("Part 25 call:")
print("    part11.dice_from_prediction(logits, mask)")
print("=" * 70)