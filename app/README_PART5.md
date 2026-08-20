# Phase 3 - Part 5: Grad-CAM XAI Dashboard

## Purpose

Integrate the previously validated Grad-CAM concept into the
deployment dashboard without retraining or changing model weights.

## XAI pipeline

```text
Existing dashboard tensor
        ↓
Existing trained SwinClassifier
        ↓
Forward features
        ↓
[1, 7, 7, 768]
        ↓
Target predicted class
        ↓
Gradient of target score
        ↓
Global-average gradient weights
        ↓
Weighted feature map
        ↓
ReLU
        ↓
Normalize
        ↓
Grad-CAM heatmap
        ↓
MRI overlay
```

## Safety check

Before creating the heatmap, the XAI engine reconstructs the
feature-to-classifier path and compares its logits with the model's
normal forward output.

If the maximum logit error exceeds `1e-4`, XAI stops and generates
no heatmap.

This prevents the dashboard from silently explaining the wrong
computation graph.

## No training

Part 5 performs:

- inference
- gradient computation for explanation
- image visualization

It does NOT perform:

- optimizer steps
- backward parameter updates
- retraining
- checkpoint saving
- weight modification

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

Workflow:

1. MRI Analysis
2. Upload DICOM series
3. Analyze MRI
4. Open Explainable AI
5. Generate XAI Explanation

## Expected feature representation

The project has previously validated:

```text
[1, 7, 7, 768]
```

The XAI engine expects a spatial Swin feature representation and
refuses to guess if the current model exposes a different structure.

## Outputs

Generated files are written under:

```text
outputs/xai_dashboard/
```

including:

```text
heatmaps/
overlays/
```

## Interpretation

A Grad-CAM overlay indicates regions associated with the model's
prediction. It is not a segmentation mask and does not prove that
a particular anatomical structure contains disease.
