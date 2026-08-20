# Phase 3 - Part 6: Professional Results Dashboard

## Purpose

Consolidate the complete inference workflow into a single
submission-ready results page.

## Included

- MRI three-slice viewer
- predicted class
- confidence
- class probabilities
- Grad-CAM heatmap
- Grad-CAM overlay
- XAI statistics
- DICOM series information
- selected channel information
- tensor/input verification
- research-use notice

## Data flow

```text
DICOM
  ↓
Part 3 DICOM engine
  ↓
Part 4 MRI viewer
  ↓
Existing trained model
  ↓
Prediction
  ↓
Part 5 Grad-CAM
  ↓
Part 6 Professional Results
```

## No training

This part does not:

- retrain the model
- modify model weights
- save checkpoints
- use an optimizer

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

## Test workflow

1. Open MRI Analysis.
2. Upload one DICOM series.
3. Analyze the MRI.
4. Open Explainable AI.
5. Generate Grad-CAM.
6. Open Professional Results.

The Results page consolidates the current session's prediction
and XAI information.

## Next stage

Part 7 will use the compact `report_summary` stored in Streamlit
session state to create a professional downloadable project report.
