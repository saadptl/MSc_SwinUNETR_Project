# Phase 3 - Part 7: Professional PDF Report

## Purpose

Turn a completed dashboard inference session into a structured,
submission-ready PDF report.

## Included sections

1. Executive summary
2. MRI input and slice selection
3. Model prediction
4. Class probabilities
5. Model input verification
6. Grad-CAM XAI
7. Technical summary
8. Research limitations and disclaimer

## XAI behavior

If Grad-CAM has been generated in the current dashboard session,
the report includes the saved heatmap and overlay.

If XAI has not been generated, the report clearly states that
the visualization is unavailable instead of inventing one.

## No retraining

The report generator only reads inference-session outputs.

It does not:

- modify model weights
- modify checkpoints
- retrain the model
- run optimizer steps

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

Then:

```text
MRI Analysis
    ↓
Analyze MRI
    ↓
Explainable AI
    ↓
Generate Grad-CAM
    ↓
Professional Results
    ↓
Report Generation
    ↓
Generate Professional PDF Report
```

## Output

The generated report is saved under:

```text
outputs/reports/lumbar_spine_ai_analysis_report.pdf
```

The dashboard also provides a download button.
