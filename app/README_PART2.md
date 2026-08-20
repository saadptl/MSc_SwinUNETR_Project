# Phase 3 - Part 2: Existing Model Integration

## Purpose

Connect the professional dashboard to the already-trained
`models/best_model.pth`.

## No training

This stage is inference-only:

- no optimizer
- no backward pass
- no epochs
- no retraining
- no checkpoint writing
- no checkpoint modification

## Run

From the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app\main.py
```

Open:

**MRI Analysis → Load Trained Model**

Then run:

**Inference Engine Test**

The test uses a zero tensor only to verify the model's forward-pass
compatibility. It is not a medical prediction.

## Expected architecture

- SwinClassifier
- 3 output classes
- 768 classifier input features
- 224 x 224 input
- 3 input channels

## Expected checkpoint

```text
models/best_model.pth
```
