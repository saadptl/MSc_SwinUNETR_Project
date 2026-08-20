# Phase 3 - Part 1: Dashboard Foundation

This directory contains the first clean dashboard foundation.

## Run

From the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

If Streamlit is not installed:

```powershell
.\venv\Scripts\python.exe -m pip install streamlit
```

## Current status

Part 1 provides:

- professional navigation
- Home/Dashboard page
- MRI Analysis placeholder
- Results page
- XAI page
- Report page
- About page
- centralized configuration
- dashboard styling
- session-state foundation

## Not yet connected

The following are intentionally added in later phases:

- DICOM series discovery
- three-slice construction
- trained-model inference
- Grad-CAM inference
- PDF report generation

No training code belongs in this dashboard.
