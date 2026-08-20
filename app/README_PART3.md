# Phase 3 - Part 3: DICOM & MRI Processing Engine

## Purpose

Connect the dashboard to real DICOM MRI data and create the same
three-channel input convention used by the trained classifier.

## Pipeline

```text
DICOM upload
    ↓
Read series
    ↓
Sort slices
    ↓
Choose middle slice
    ↓
Previous / Middle / Next
    ↓
Min-max normalization
    ↓
CLAHE
    ↓
Resize 224 × 224
    ↓
/255
    ↓
Tensor [1, 3, 224, 224]
    ↓
Existing trained model
```

## Important

This stage is inference-only.

It does not:

- train
- retrain
- modify model weights
- modify checkpoints
- use an optimizer

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

Then:

1. Open **MRI Analysis**
2. Upload DICOM files from one series
3. Select the middle slice
4. Review Channel 0 / 1 / 2
5. Click **Analyze MRI**
6. Review prediction and probabilities

## Input convention

The dashboard constructs:

```text
Channel 0 = previous slice
Channel 1 = selected middle slice
Channel 2 = next slice
```

with final tensor:

```text
[1, 3, 224, 224]
```

## Edge cases

The loader handles:

- one-slice series
- two-slice series
- normal multi-slice series
- missing InstanceNumber
- ImagePositionPatient ordering when available
- MONOCHROME1 images
- unreadable DICOM files
