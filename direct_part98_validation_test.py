import sys
import importlib.util
import torch

sys.path.insert(0, "src")

# ------------------------------------------------------------------
# Load Part11
# ------------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "part11",
    "src/segmentation_rsna_part11_controlled_pilot_training_corrected.py",
)

part11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(part11)

import segmentation_rsna_part9_3d_dataset_loader as part9


# ------------------------------------------------------------------
# Load Part98 crop functions
# ------------------------------------------------------------------

spec98 = importlib.util.spec_from_file_location(
    "part98",
    "src/segmentation_rsna_part98_strong_full_cohort_training.py",
)

part98 = importlib.util.module_from_spec(spec98)
spec98.loader.exec_module(part98)


# ------------------------------------------------------------------
# Exact Part15 validation case
# ------------------------------------------------------------------

row = {
    "study_id": 961169106,
    "series_id": 1260127192,
    "series_description": "Axial T2",
    "dicom_dir": (
        r"C:\Saad\Msc Major Project Swin Unetr Framework"
        r"\MSc_SwinUNETR_Project\dataset"
        r"\rsna-2024-lumbar-spine-degenerative-classification"
        r"\train_images\961169106\1260127192"
    ),
    "pseudo_mask_path": (
        r"C:\Saad\Msc Major Project Swin Unetr Framework"
        r"\MSc_SwinUNETR_Project\outputs"
        r"\segmentation\rsna_part6_pseudomask_generation"
        r"\pseudo_masks"
        r"\961169106_1260127192_Axial_T2.npz"
    ),
}


print("=" * 70)
print("PART98 ORIGINAL-CROP VS FULL-VOLUME TEST")
print("=" * 70)


# ------------------------------------------------------------------
# Load exact Part11 representation
# ------------------------------------------------------------------

image, target, *_ = part11.load_tensor_case(row, part9)

image = part98.normalize_image_tensor(image)
target = part98.normalize_mask_tensor(target)

image = part98.normalize_mri(image)

image = part98.resize_3d(
    image,
    part98.FULL_SHAPE,
)

target = part98.resize_mask_3d(
    target,
    part98.FULL_SHAPE,
)


print("\nFULL PREPROCESSED VOLUME")
print("-" * 70)
print("Image :", tuple(image.shape))
print("Mask  :", tuple(target.shape))
print("Target foreground:", int(torch.count_nonzero(target)))


# ------------------------------------------------------------------
# Reproduce EXACT Part98 foreground-centered crop
# ------------------------------------------------------------------

center = part98.compute_foreground_center(target)

print("\nFOREGROUND CENTER")
print("-" * 70)
print("Center:", center)

crop_image, crop_target = part98.crop_3d(
    image,
    target,
    part98.CROP_SHAPE,
    center,
)


print("\nORIGINAL PART98 CROP")
print("-" * 70)
print("Crop image :", tuple(crop_image.shape))
print("Crop mask  :", tuple(crop_target.shape))
print(
    "Crop foreground:",
    int(torch.count_nonzero(crop_target)),
)
print(
    "Crop FG ratio:",
    float(torch.count_nonzero(crop_target))
    / crop_target.numel(),
)


# ------------------------------------------------------------------
# Create exact model
# ------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", device)

model = part11.create_model(device)


# ------------------------------------------------------------------
# Load Part104
# ------------------------------------------------------------------

checkpoint_path = (
    r"outputs\segmentation\rsna_part104_final_checkpoint_selection"
    r"\checkpoints\final_segmentation_model.pth"
)

state = torch.load(
    checkpoint_path,
    map_location=device,
)

if isinstance(state, dict):
    state = state.get(
        "state_dict",
        state.get("model_state_dict", state),
    )

model.load_state_dict(
    state,
    strict=True,
)

model.eval()

print("Checkpoint load: PASS")


# ------------------------------------------------------------------
# Run ORIGINAL Part98 crop
# ------------------------------------------------------------------

crop_input = crop_image.unsqueeze(0).unsqueeze(0).float().to(device)

print("\nCROP MODEL INPUT")
print("-" * 70)
print("Input:", tuple(crop_input.shape))


with torch.no_grad():
    crop_logits = model(crop_input)
    crop_prediction = torch.argmax(
        crop_logits,
        dim=1,
    )[0].cpu()


print("\nCROP PREDICTION")
print("-" * 70)
print(
    "Unique classes:",
    torch.unique(crop_prediction).tolist(),
)

print(
    "Pred foreground:",
    int(torch.count_nonzero(crop_prediction)),
)

print(
    "Target foreground:",
    int(torch.count_nonzero(crop_target)),
)

print(
    "Pred FG ratio:",
    float(torch.count_nonzero(crop_prediction))
    / crop_prediction.numel(),
)

print(
    "Target FG ratio:",
    float(torch.count_nonzero(crop_target))
    / crop_target.numel(),
)


print("\nCROP CLASS COUNTS")
print("-" * 70)

for c in range(6):

    pred_count = int(
        (crop_prediction == c).sum()
    )

    target_count = int(
        (crop_target == c).sum()
    )

    print(
        f"Class {c}: "
        f"pred={pred_count:7d}   "
        f"target={target_count:7d}"
    )


# ------------------------------------------------------------------
# Also test FULL volume directly
# ------------------------------------------------------------------

full_input = image.unsqueeze(0).float().to(device)

print("\nFULL-VOLUME MODEL INPUT")
print("-" * 70)
print("Input:", tuple(full_input.shape))


with torch.no_grad():

    full_logits = model(full_input)

    full_prediction = torch.argmax(
        full_logits,
        dim=1,
    )[0].cpu()


print("\nFULL-VOLUME PREDICTION")
print("-" * 70)

print(
    "Pred foreground:",
    int(torch.count_nonzero(full_prediction)),
)

print(
    "Target foreground:",
    int(torch.count_nonzero(target)),
)

print(
    "Pred FG ratio:",
    float(torch.count_nonzero(full_prediction))
    / full_prediction.numel(),
)

print(
    "Target FG ratio:",
    float(torch.count_nonzero(target))
    / target.numel(),
)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)