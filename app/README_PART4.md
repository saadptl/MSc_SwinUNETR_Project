# Phase 3 - Part 4: Professional MRI Viewer

## What this adds

The dashboard now presents:

- three preprocessed MRI slices
- previous / middle / next channel roles
- model prediction
- confidence
- class probabilities
- DICOM series count
- selected slice metadata
- tensor shape
- preprocessing summary

## Input construction

```text
Channel 0 → Previous
Channel 1 → Middle
Channel 2 → Next
```

Final tensor:

```text
[1, 3, 224, 224]
```

## Run

From the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

Then:

1. Open MRI Analysis.
2. Upload one DICOM series.
3. Select the middle slice.
4. Click Analyze Selected MRI.
5. Review the three-slice viewer.
6. Open Results for the detailed prediction.

## No retraining

This part is inference-only and does not modify:

- model weights
- checkpoint files
- optimizer state
- training configuration
