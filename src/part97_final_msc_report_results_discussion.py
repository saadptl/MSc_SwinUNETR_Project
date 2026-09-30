from pathlib import Path
import json
import csv
from datetime import datetime


# =============================================================================
# PART 97 — FINAL MSc PROJECT REPORT & RESULTS DISCUSSION PACKAGE
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "outputs" / "final" / "part97_msc_report_package"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR = OUTPUT_DIR / "tables"
CSV_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# VERIFIED PROJECT RESULTS
# =============================================================================

PROJECT_TITLE = (
    "Explainable Swin-UNETR Framework for Automated Lumbar Spine "
    "Disease Detection and Classification from MRI Images"
)

SEGMENTATION = {
    "model": "Swin-UNETR",
    "checkpoint": (
        "outputs/segmentation/rsna_part84_final_reproducible_training/"
        "checkpoints/part84_best_model.pth"
    ),
    "parameters": 4078116,
    "training_epochs": 8,
    "validation_studies": 50,
    "global_foreground_dice": 0.193388,
    "mean_case_foreground_dice": 0.193388,
    "median_case_foreground_dice": 0.203689,
    "macro_foreground_dice": 0.193388,
    "status": "FINAL REPRODUCIBLE",
}

CLASSIFICATION_BASELINE = {
    "part": "Part90",
    "model": "Lightweight3DCNN",
    "parameters": 77907,
    "validation_studies": 395,
    "accuracy": 0.775389,
    "majority_baseline": 0.775389,
    "balanced_accuracy": 0.333333,
    "moderate_recall": 0.0,
    "severe_recall": 0.0,
}

CLASSIFICATION_FINAL = {
    "part": "Part92",
    "model": "Part92CNN",
    "parameters": 20083,
    "training_epochs": 5,
    "selected_epoch": 1,
    "validation_studies": 395,
    "accuracy": 0.716626,
    "majority_baseline": 0.775389,
    "balanced_accuracy": 0.438530,
    "macro_f1": 0.442275,
    "moderate_recall": 0.360377,
    "severe_recall": 0.115894,
    "normal_mild_recall": 0.904806,
    "status": "FINAL CLASSIFIER",
}

INTEGRATED = {
    "validation_studies": 395,
    "successful": 395,
    "failed": 0,
    "success_rate": 1.0,
    "mean_segmentation_fg_fraction": 0.071236,
    "mean_segmentation_fg_voxels": 42016.76,
    "mean_classifier_confidence": 0.480336,
    "mean_severe_probability": 0.240634,
}

EXPLAINABILITY = {
    "validation_studies": 395,
    "successful": 395,
    "failed": 0,
    "success_rate": 1.0,
    "visualizations": 10,
    "mean_foreground_fraction": 0.069968,
    "mean_classifier_confidence": 0.480336,
    "mean_severe_probability": 0.240634,
}

REPRODUCIBILITY = {
    "Part84 final segmentation checkpoint": {
        "status": "REPRODUCIBLE",
        "evidence": "Frozen checkpoint exists and Part85 independently reloaded it.",
    },
    "Part85 segmentation audit": {
        "status": "REPRODUCIBLE",
        "evidence": "50 validation studies evaluated with frozen Part84 checkpoint.",
    },
    "Part90 baseline classification": {
        "status": "REPRODUCIBLE",
        "evidence": "Part93 strict-loaded and evaluated Part90 checkpoint.",
    },
    "Part92 imbalance-aware classification": {
        "status": "REPRODUCIBLE",
        "evidence": "Part93 strict-loaded and evaluated Part92 checkpoint.",
    },
    "Part94 integrated inference": {
        "status": "VALIDATED",
        "evidence": "395/395 validation studies processed successfully.",
    },
    "Part95 explainability": {
        "status": "VALIDATED",
        "evidence": (
            "395/395 validation studies processed and 10 representative "
            "visualizations generated."
        ),
    },
    "Historical Part71 Dice = 0.044026": {
        "status": "NON-REPRODUCIBLE",
        "evidence": (
            "Parts79–83 could not reconstruct the historical value from "
            "the saved checkpoint."
        ),
    },
}


# =============================================================================
# TABLE DATA
# =============================================================================

FINAL_RESULTS_ROWS = [
    [
        "Segmentation",
        "Swin-UNETR",
        "Part15 validation subset",
        50,
        "Global foreground Dice",
        0.193388,
        "Median-case foreground Dice",
        0.203689,
        "Final reproducible",
    ],
    [
        "Classification",
        "Part90 baseline",
        "Part87 validation",
        395,
        "Accuracy",
        0.775389,
        "Majority baseline",
        0.775389,
        "Baseline",
    ],
    [
        "Classification",
        "Part90 baseline",
        "Part87 validation",
        395,
        "Balanced accuracy",
        0.333333,
        "Severe recall",
        0.0,
        "Baseline",
    ],
    [
        "Classification",
        "Part92 imbalance-aware",
        "Part87 validation",
        395,
        "Accuracy",
        0.716626,
        "Balanced accuracy",
        0.438530,
        "Final classifier",
    ],
    [
        "Classification",
        "Part92 imbalance-aware",
        "Part87 validation",
        395,
        "Macro F1",
        0.442275,
        "Severe recall",
        0.115894,
        "Final classifier",
    ],
    [
        "Integrated inference",
        "Part94",
        "Part87 validation",
        395,
        "Pipeline success rate",
        1.0,
        "Mean classifier confidence",
        0.480336,
        "Validated",
    ],
    [
        "Explainability",
        "Part95",
        "Part87 validation",
        395,
        "Explainable inference success rate",
        1.0,
        "Visualizations",
        10,
        "Validated",
    ],
]

CLASSIFICATION_COMPARISON_ROWS = [
    [
        "Accuracy",
        CLASSIFICATION_BASELINE["accuracy"],
        CLASSIFICATION_FINAL["accuracy"],
        CLASSIFICATION_FINAL["accuracy"]
        - CLASSIFICATION_BASELINE["accuracy"],
    ],
    [
        "Balanced accuracy",
        CLASSIFICATION_BASELINE["balanced_accuracy"],
        CLASSIFICATION_FINAL["balanced_accuracy"],
        CLASSIFICATION_FINAL["balanced_accuracy"]
        - CLASSIFICATION_BASELINE["balanced_accuracy"],
    ],
    [
        "Moderate recall",
        CLASSIFICATION_BASELINE["moderate_recall"],
        CLASSIFICATION_FINAL["moderate_recall"],
        CLASSIFICATION_FINAL["moderate_recall"]
        - CLASSIFICATION_BASELINE["moderate_recall"],
    ],
    [
        "Severe recall",
        CLASSIFICATION_BASELINE["severe_recall"],
        CLASSIFICATION_FINAL["severe_recall"],
        CLASSIFICATION_FINAL["severe_recall"]
        - CLASSIFICATION_BASELINE["severe_recall"],
    ],
]


# =============================================================================
# HELPERS
# =============================================================================

def pct(value):
    return f"{value * 100:.2f}%"


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_text(path, text):
    path.write_text(text, encoding="utf-8")


# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

executive_summary = f"""
PROJECT TITLE
{PROJECT_TITLE}

FINAL PROJECT RESULTS SUMMARY
================================

The project developed an explainable MRI-based lumbar spine analysis
framework using Swin-UNETR for three-dimensional segmentation and a
separate imbalance-aware classification model for lumbar spine
degenerative disease severity classification.

The final reproducible segmentation model is the Part84 Swin-UNETR
checkpoint. It contains {SEGMENTATION["parameters"]:,} trainable parameters
and achieved a global foreground Dice of
{SEGMENTATION["global_foreground_dice"]:.6f} on the audited 50-study
validation subset.

The final classification model is the Part92 imbalance-aware classifier.
On the 395-study validation cohort it achieved an accuracy of
{CLASSIFICATION_FINAL["accuracy"]:.6f}, balanced accuracy of
{CLASSIFICATION_FINAL["balanced_accuracy"]:.6f}, and macro F1 of
{CLASSIFICATION_FINAL["macro_f1"]:.6f}. Moderate-class recall was
{CLASSIFICATION_FINAL["moderate_recall"]:.6f}, while Severe-class recall
was {CLASSIFICATION_FINAL["severe_recall"]:.6f}.

The Part90 baseline achieved an accuracy of
{CLASSIFICATION_BASELINE["accuracy"]:.6f}, exactly matching the majority
class baseline of {CLASSIFICATION_BASELINE["majority_baseline"]:.6f}.
It had zero recall for both Moderate and Severe classes. Therefore,
the Part90 headline accuracy does not represent useful balanced
severity classification performance.

The integrated Part94 inference pipeline successfully processed all
395 validation studies without failure. Part95 similarly processed all
395 studies and generated 10 representative explainability
visualizations.

The historical Part71 Dice value of 0.044026 was investigated in
Parts79–83 but could not be reconstructed from the saved checkpoint.
It is therefore excluded from the final reproducible performance
claims.

Overall, the project demonstrates a technically functional and
reproducible research pipeline, while the quantitative results also
show that further model development would be required before clinical
deployment or clinical performance claims.
"""


# =============================================================================
# METHODOLOGY
# =============================================================================

methodology = """
FINAL METHODOLOGY SUMMARY
=========================

1. DATASET
The project uses the RSNA 2024 Lumbar Spine Degenerative Classification
dataset.

The classification dataset contains 1,975 studies and 25 severity
targets covering five lumbar conditions across five lumbar levels.

2. STUDY-LEVEL SPLITTING
A leakage-free study-level split was created in Part87:

Training studies : 1,580
Validation studies : 395
Study overlap : 0

The split was created at the study level rather than at individual
image level to reduce the possibility of patient/study leakage.

3. MRI DATA PIPELINE
The MRI loader was validated in Part88 and the complete dataset
pipeline was validated in Part89.

The classification pipeline produces tensors with the expected
three-dimensional format and masks missing target values rather than
incorrectly treating missing labels as Normal/Mild.

4. SEGMENTATION
Swin-UNETR was used as the three-dimensional segmentation architecture.

The final reproducible segmentation configuration used:
- RSNA-derived pseudo-masks
- six output classes
- centered spatial crops
- 32 x 64 x 64 training crops
- class-balanced Dice + weighted cross-entropy
- RTX 2050 CUDA execution
- eight training epochs in the final reproducible run

The final Part84 checkpoint was independently reloaded and audited
in Part85.

5. CLASSIFICATION
Part90 established the controlled baseline classification experiment.

The baseline showed severe majority-class behavior.

Part92 introduced class-imbalance-aware training. The selected Part92
checkpoint was the best epoch according to macro F1 during its
controlled training experiment.

Important methodological limitation:
Part92 was not architecturally identical to the actual Part90 model.
Part90 used the Lightweight3DCNN architecture with 77,907 parameters,
while Part92 used Part92CNN with 20,083 parameters.

Therefore, the Part90-to-Part92 comparison demonstrates that the
imbalance-aware experiment produced substantially more minority-class
signal, but it cannot be interpreted as a perfectly isolated
one-variable class-weighting ablation.

6. INTEGRATED PIPELINE
Part94 connected the frozen segmentation and classification models
into a sequential inference workflow.

The classifier and segmentation model were trained separately.
The classifier does not consume Swin-UNETR internal segmentation
features.

Therefore, the segmentation output should be interpreted as
spatial/output-level evidence accompanying the classification result,
not as proof that the classifier directly used the highlighted
segmentation regions.

7. EXPLAINABILITY
Part95 generated representative visualization outputs containing
MRI slices, segmentation overlays, severity distributions and
classification probability information.

This is output-level/spatial explainability. It is not Grad-CAM,
SHAP, integrated gradients or another feature-attribution method.
"""


# =============================================================================
# SEGMENTATION DISCUSSION
# =============================================================================

segmentation_discussion = f"""
SEGMENTATION RESULTS AND DISCUSSION
===================================

The final Swin-UNETR model achieved a global foreground Dice of
{SEGMENTATION["global_foreground_dice"]:.6f} on the audited 50-study
validation subset.

The result indicates that the segmentation component learned some
foreground structure but remains limited in terms of accurate
voxel-level localization.

The model did not remain completely collapsed to an all-background
prediction in the final Part84 model. This is an important improvement
over several earlier controlled experiments in which foreground
prediction collapsed.

However, the global Dice value remains low. The segmentation target
masks are pseudo-masks derived from RSNA point/localizer annotations,
rather than manually delineated expert segmentation masks. The
extreme sparsity and mixed annotation geometry make the segmentation
task particularly difficult.

Earlier controlled experiments demonstrated that:
- the pseudo-masks are highly sparse;
- foreground classes are severely imbalanced;
- several foreground classes collapse more readily than others;
- class-balanced training can improve foreground signal;
- point-based auxiliary supervision did not provide a meaningful
  improvement;
- spatial consistency supervision did not provide a meaningful
  improvement;
- boundary supervision did not provide a meaningful improvement.

For the final report, the global foreground Dice should be treated as
the primary segmentation metric.

The segmentation result should not be described as clinically
accurate segmentation. It should instead be presented as a
research-stage segmentation component that provides spatial evidence
within the integrated framework.
"""


# =============================================================================
# CLASSIFICATION DISCUSSION
# =============================================================================

classification_discussion = f"""
CLASSIFICATION RESULTS AND DISCUSSION
=====================================

The Part90 baseline achieved:

Accuracy             : {CLASSIFICATION_BASELINE["accuracy"]:.6f}
Majority baseline    : {CLASSIFICATION_BASELINE["majority_baseline"]:.6f}
Balanced accuracy    : {CLASSIFICATION_BASELINE["balanced_accuracy"]:.6f}
Moderate recall      : {CLASSIFICATION_BASELINE["moderate_recall"]:.6f}
Severe recall        : {CLASSIFICATION_BASELINE["severe_recall"]:.6f}

Because the Part90 accuracy exactly matched the majority-class baseline,
the 77.54% accuracy should not be interpreted as strong classification
performance.

The imbalance-aware Part92 experiment produced:

Accuracy             : {CLASSIFICATION_FINAL["accuracy"]:.6f}
Balanced accuracy    : {CLASSIFICATION_FINAL["balanced_accuracy"]:.6f}
Macro F1              : {CLASSIFICATION_FINAL["macro_f1"]:.6f}
Moderate recall       : {CLASSIFICATION_FINAL["moderate_recall"]:.6f}
Severe recall         : {CLASSIFICATION_FINAL["severe_recall"]:.6f}

Compared with Part90, balanced accuracy increased from
{CLASSIFICATION_BASELINE["balanced_accuracy"]:.6f} to
{CLASSIFICATION_FINAL["balanced_accuracy"]:.6f}.

Moderate recall increased from 0 to
{CLASSIFICATION_FINAL["moderate_recall"]:.6f}, while Severe recall
increased from 0 to {CLASSIFICATION_FINAL["severe_recall"]:.6f}.

The raw accuracy decreased from
{CLASSIFICATION_BASELINE["accuracy"]:.6f} to
{CLASSIFICATION_FINAL["accuracy"]:.6f}.

This trade-off is expected when a model stops simply favoring the
majority class. For a highly imbalanced medical classification task,
balanced accuracy, macro F1 and per-class recall provide more useful
information than raw accuracy alone.

Nevertheless, Severe recall remains limited. Therefore, the final
classifier demonstrates non-trivial severity signal but should not
be presented as clinically reliable disease severity prediction.

IMPORTANT ARCHITECTURAL CAVEAT
------------------------------

The Part92 architecture contains 20,083 parameters and differs from
the actual Part90 architecture, which contains 77,907 parameters.

Consequently, the Part90-versus-Part92 comparison should be described
as a controlled experimental comparison demonstrating the effect of
an imbalance-aware classification approach in the developed pipeline,
but not as a mathematically isolated loss-only ablation.

This limitation should be explicitly stated in the dissertation.
"""


# =============================================================================
# INTEGRATED PIPELINE DISCUSSION
# =============================================================================

integrated_discussion = f"""
INTEGRATED PIPELINE RESULTS
===========================

Part94 successfully processed {INTEGRATED["successful"]} out of
{INTEGRATED["validation_studies"]} validation studies.

Pipeline success rate : {INTEGRATED["success_rate"]:.4f}
Mean foreground fraction : {INTEGRATED["mean_segmentation_fg_fraction"]:.6f}
Mean foreground voxels : {INTEGRATED["mean_segmentation_fg_voxels"]:.2f}
Mean classifier confidence : {INTEGRATED["mean_classifier_confidence"]:.6f}
Mean Severe probability : {INTEGRATED["mean_severe_probability"]:.6f}

The principal result of Part94 is successful end-to-end execution,
rather than a claim of 100% diagnostic accuracy.

The segmentation and classification components remain independently
trained frozen models. The pipeline demonstrates that they can be
executed sequentially for the same validation studies and their
outputs can be combined into an integrated research workflow.

The mean segmentation foreground fraction is an output statistic and
must not be interpreted as segmentation accuracy or Dice.

Similarly, mean classifier confidence is not classification accuracy.
It represents the average confidence associated with the predicted
classification outputs.
"""


# =============================================================================
# EXPLAINABILITY DISCUSSION
# =============================================================================

explainability_discussion = f"""
EXPLAINABILITY RESULTS
======================

Part95 processed all {EXPLAINABILITY["validation_studies"]} validation
studies successfully and generated {EXPLAINABILITY["visualizations"]}
representative visualizations.

Explainable inference success rate : {EXPLAINABILITY["success_rate"]:.4f}
Mean foreground fraction           : {EXPLAINABILITY["mean_foreground_fraction"]:.6f}
Mean classifier confidence         : {EXPLAINABILITY["mean_classifier_confidence"]:.6f}
Mean Severe probability            : {EXPLAINABILITY["mean_severe_probability"]:.6f}

The visualizations provide spatial/output-level evidence by showing
MRI slices together with segmentation predictions and classification
probability information.

This improves interpretability compared with reporting only a final
severity label.

However, the explainability implementation does not establish causal
feature attribution. Because the classifier was trained separately
from the Swin-UNETR model, the segmentation overlay cannot be claimed
to be the exact evidence used internally by the classifier.

Accordingly, the explainability component should be described as
visual/spatial output-level explainability.
"""


# =============================================================================
# REPRODUCIBILITY DISCUSSION
# =============================================================================

reproducibility_discussion = """
REPRODUCIBILITY AND EXPERIMENTAL AUDIT
======================================

The final project contains independently reloadable checkpoints and
evaluation artifacts.

Part84:
The final Swin-UNETR checkpoint exists and was independently reloaded
during Part85.

Part90:
The baseline classification checkpoint was strictly loaded and
evaluated in Part93.

Part92:
The final imbalance-aware classification checkpoint was strictly
loaded and evaluated in Part93.

Part94:
The integrated inference pipeline processed the complete 395-study
validation cohort successfully.

Part95:
The explainability pipeline processed the complete 395-study
validation cohort and generated representative visualizations.

A historical Part71 Dice value of 0.044026 was subjected to several
forensic and reproducibility audits in Parts79–83. The value could
not be reconstructed from the saved checkpoint.

Therefore, 0.044026 is explicitly classified as NON-REPRODUCIBLE and
is excluded from the final reproducible performance claims.

This decision strengthens the scientific integrity of the final
project results because only independently verified and reproducible
results are used as final claims.
"""


# =============================================================================
# LIMITATIONS
# =============================================================================

limitations = """
PROJECT LIMITATIONS
===================

1. PSEUDO-SEGMENTATION TARGETS
The segmentation masks are pseudo-masks derived from RSNA point/localizer
annotations rather than expert manual voxel-level segmentation.

2. EXTREME FOREGROUND SPARSITY
The segmentation task contains highly sparse foreground targets and
substantial class imbalance.

3. LIMITED SEGMENTATION PERFORMANCE
The final global foreground Dice remains low and therefore the
segmentation model should be considered a research-stage component.

4. CLASSIFICATION IMBALANCE
The disease severity labels are strongly imbalanced. The baseline
accuracy was shown to be misleading because it matched the majority
class baseline.

5. LIMITED SEVERE-CLASS RECALL
The final classifier achieved only 0.115894 Severe recall, indicating
that severe cases remain difficult to identify reliably.

6. ARCHITECTURAL DIFFERENCE
Part90 and Part92 do not use identical architectures. Part90 has
77,907 parameters while Part92 has 20,083 parameters. Therefore,
their comparison cannot be treated as a perfectly isolated
single-variable class-weighting experiment.

7. SEPARATE TRAINING
The segmentation and classification models were trained independently.
The classifier does not directly consume Swin-UNETR segmentation
features.

8. OUTPUT-LEVEL EXPLAINABILITY
The explainability component uses spatial/output-level visualization
rather than feature attribution such as Grad-CAM or SHAP.

9. VALIDATION-SUBSET DIFFERENCE
The final segmentation audit uses a 50-study validation subset,
whereas classification and integrated inference use the 395-study
classification validation cohort.

10. NO CLINICAL VALIDATION
The project is a research prototype. The reported results do not
constitute clinical validation, medical-device validation, or a
recommendation for clinical deployment.

11. DATASET SCOPE
The experiments are based on the RSNA dataset used in the project.
External multi-institutional validation has not been performed.

12. COMPUTATIONAL LIMITATIONS
Training and inference were conducted on an NVIDIA RTX 2050 with
4 GB VRAM, which constrained model size, batch size and experimental
scope.
"""


# =============================================================================
# FUTURE WORK
# =============================================================================

future_work = """
FUTURE WORK
===========

The following improvements are recommended for future research:

1. Replace pseudo-masks with expert-annotated segmentation masks.

2. Increase the size and diversity of the segmentation training cohort.

3. Develop a stronger 3D segmentation architecture or optimize
   Swin-UNETR with larger-scale training when computational resources
   permit.

4. Investigate improved sampling strategies for rare disease regions.

5. Use a single consistent classification architecture when comparing
   loss functions or class-balancing methods.

6. Investigate focal loss, asymmetric loss or calibrated
   class-balanced objectives.

7. Perform probability calibration and threshold optimization on a
   dedicated validation set.

8. Add stronger patient/study-level external validation.

9. Investigate true feature-level explainability such as Grad-CAM,
   integrated gradients or SHAP-style analysis where technically
   appropriate.

10. Develop a genuinely joint architecture in which segmentation
    representations contribute directly to disease classification.

11. Evaluate the final framework using external clinical datasets.

12. Conduct statistical confidence intervals and repeated
    cross-validation when sufficient computational resources are
    available.
"""


# =============================================================================
# FINAL CONCLUSION
# =============================================================================

conclusion = f"""
FINAL CONCLUSION
================

This project developed an explainable research framework for automated
lumbar spine MRI analysis using three-dimensional Swin-UNETR
segmentation and an imbalance-aware severity classification model.

The final reproducible Swin-UNETR model achieved a global foreground
Dice of {SEGMENTATION["global_foreground_dice"]:.6f} on the audited
segmentation validation subset.

The final Part92 classification model achieved an accuracy of
{CLASSIFICATION_FINAL["accuracy"]:.6f}, balanced accuracy of
{CLASSIFICATION_FINAL["balanced_accuracy"]:.6f}, and macro F1 of
{CLASSIFICATION_FINAL["macro_f1"]:.6f}. Its Moderate recall was
{CLASSIFICATION_FINAL["moderate_recall"]:.6f} and Severe recall was
{CLASSIFICATION_FINAL["severe_recall"]:.6f}.

The comparison with the Part90 baseline demonstrated why accuracy
alone is inadequate for this imbalanced medical classification task.
The baseline accuracy of 0.775389 exactly matched the majority-class
baseline and had zero recall for Moderate and Severe classes.

The imbalance-aware experiment produced substantially more
minority-class signal, although its overall classification performance
remains limited.

The integrated inference system successfully processed all 395
classification validation studies, and the explainability pipeline
also processed all 395 studies while generating representative
visualizations.

The project therefore demonstrates a complete, reproducible research
pipeline from MRI preprocessing through segmentation, severity
classification, integrated inference and output-level explainability.

At the same time, the quantitative results indicate that the current
system should be regarded as an MSc research prototype rather than a
clinically deployable diagnostic system.

A major strength of the final project is the explicit reproducibility
audit. The historical Part71 Dice value of 0.044026 could not be
reconstructed from its saved checkpoint and was therefore excluded
from the final performance claims. This ensures that the reported
final results are based on independently verified artifacts.
"""


# =============================================================================
# VIVA / PRESENTATION SUMMARY
# =============================================================================

viva_summary = """
VIVA / PRESENTATION KEY POINTS
==============================

PROJECT AIM
-----------
Develop an explainable deep-learning framework for automated lumbar
spine MRI disease segmentation and severity classification.

WHY SWIN-UNETR?
---------------
Swin-UNETR provides a transformer-based 3D architecture suitable for
learning spatial relationships in volumetric medical images.

WHY SEGMENTATION?
-----------------
Segmentation provides spatial localization of relevant lumbar spine
regions and supports interpretable visual outputs.

WHY CLASS-BALANCED TRAINING?
----------------------------
The RSNA classification targets are strongly imbalanced. The baseline
model achieved majority-class behavior, making raw accuracy misleading.

WHAT WAS THE BASELINE?
----------------------
Part90:
Accuracy = 0.775389
Balanced accuracy = 0.333333
Moderate recall = 0
Severe recall = 0

WHAT WAS THE FINAL CLASSIFIER?
------------------------------
Part92:
Accuracy = 0.716626
Balanced accuracy = 0.438530
Macro F1 = 0.442275
Moderate recall = 0.360377
Severe recall = 0.115894

WHAT WAS THE FINAL SEGMENTATION RESULT?
---------------------------------------
Global foreground Dice = 0.193388

DID THE COMPLETE PIPELINE WORK?
--------------------------------
Yes.
395/395 integrated studies processed successfully.

DID EXPLAINABILITY WORK?
------------------------
Yes.
395/395 studies processed successfully and 10 representative
visualizations were generated.

IS THE MODEL CLINICALLY READY?
------------------------------
No.
The project is a research prototype. Segmentation performance,
Severe-class recall, external validation and clinical validation
remain insufficient for clinical deployment.

WHAT IS THE MOST IMPORTANT LIMITATION?
--------------------------------------
The segmentation targets are pseudo-masks derived from point/localizer
annotations, and the final segmentation Dice remains low.

WHAT IS AN IMPORTANT EXPERIMENTAL LIMITATION?
---------------------------------------------
Part90 and Part92 use different architectures, so their comparison
cannot be interpreted as a perfectly isolated class-weighting ablation.

WHY WAS THE PART71 RESULT REMOVED?
-----------------------------------
The historical Dice value of 0.044026 could not be reproduced from
the saved checkpoint during Parts79–83, so it was excluded from the
final reproducible claims.

WHAT IS THE MAIN CONTRIBUTION?
------------------------------
A complete and reproducible research pipeline combining 3D MRI
segmentation, imbalance-aware disease severity classification,
integrated inference and spatial/output-level explainability.
"""


# =============================================================================
# COMPLETE REPORT
# =============================================================================

complete_report = f"""
================================================================================
MSc COMPUTER SCIENCE / DATA SCIENCE MAJOR PROJECT
FINAL RESULTS, DISCUSSION AND CONCLUSION
================================================================================

PROJECT TITLE
{PROJECT_TITLE}

Generated by Part97
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

================================================================================
1. EXECUTIVE SUMMARY
================================================================================

{executive_summary.strip()}

================================================================================
2. FINAL METHODOLOGY
================================================================================

{methodology.strip()}

================================================================================
3. FINAL QUANTITATIVE RESULTS
================================================================================

SEGMENTATION
------------
Global foreground Dice       : {SEGMENTATION["global_foreground_dice"]:.6f}
Median-case foreground Dice  : {SEGMENTATION["median_case_foreground_dice"]:.6f}
Validation studies           : {SEGMENTATION["validation_studies"]}
Parameters                   : {SEGMENTATION["parameters"]:,}

CLASSIFICATION BASELINE — PART90
---------------------------------
Accuracy                     : {CLASSIFICATION_BASELINE["accuracy"]:.6f}
Majority baseline            : {CLASSIFICATION_BASELINE["majority_baseline"]:.6f}
Balanced accuracy            : {CLASSIFICATION_BASELINE["balanced_accuracy"]:.6f}
Moderate recall              : {CLASSIFICATION_BASELINE["moderate_recall"]:.6f}
Severe recall                : {CLASSIFICATION_BASELINE["severe_recall"]:.6f}

FINAL CLASSIFIER — PART92
-------------------------
Accuracy                     : {CLASSIFICATION_FINAL["accuracy"]:.6f}
Balanced accuracy            : {CLASSIFICATION_FINAL["balanced_accuracy"]:.6f}
Macro F1                     : {CLASSIFICATION_FINAL["macro_f1"]:.6f}
Moderate recall              : {CLASSIFICATION_FINAL["moderate_recall"]:.6f}
Severe recall                : {CLASSIFICATION_FINAL["severe_recall"]:.6f}

INTEGRATED INFERENCE — PART94
-----------------------------
Validation studies           : {INTEGRATED["validation_studies"]}
Successful studies           : {INTEGRATED["successful"]}
Failed studies               : {INTEGRATED["failed"]}
Success rate                 : {INTEGRATED["success_rate"]:.4f}

EXPLAINABILITY — PART95
-----------------------
Validation studies           : {EXPLAINABILITY["validation_studies"]}
Successful studies           : {EXPLAINABILITY["successful"]}
Failed studies               : {EXPLAINABILITY["failed"]}
Visualizations               : {EXPLAINABILITY["visualizations"]}
Success rate                 : {EXPLAINABILITY["success_rate"]:.4f}

================================================================================
4. SEGMENTATION DISCUSSION
================================================================================

{segmentation_discussion.strip()}

================================================================================
5. CLASSIFICATION DISCUSSION
================================================================================

{classification_discussion.strip()}

================================================================================
6. INTEGRATED PIPELINE DISCUSSION
================================================================================

{integrated_discussion.strip()}

================================================================================
7. EXPLAINABILITY DISCUSSION
================================================================================

{explainability_discussion.strip()}

================================================================================
8. REPRODUCIBILITY
================================================================================

{reproducibility_discussion.strip()}

================================================================================
9. LIMITATIONS
================================================================================

{limitations.strip()}

================================================================================
10. FUTURE WORK
================================================================================

{future_work.strip()}

================================================================================
11. FINAL CONCLUSION
================================================================================

{conclusion.strip()}

================================================================================
12. VIVA / PRESENTATION SUMMARY
================================================================================

{viva_summary.strip()}

================================================================================
END OF PART97 REPORT
================================================================================
"""


# =============================================================================
# SAVE MAIN REPORT
# =============================================================================

report_txt = OUTPUT_DIR / "part97_final_msc_report.txt"
write_text(report_txt, complete_report)

report_md = OUTPUT_DIR / "part97_final_msc_report.md"
write_text(report_md, complete_report)


# =============================================================================
# SAVE FINAL RESULTS CSV
# =============================================================================

write_csv(
    CSV_DIR / "part97_final_results_table.csv",
    [
        "Category",
        "Component",
        "Dataset Split",
        "Studies",
        "Primary Metric",
        "Value",
        "Secondary Metric",
        "Secondary Value",
        "Status",
    ],
    FINAL_RESULTS_ROWS,
)


# =============================================================================
# SAVE CLASSIFICATION COMPARISON CSV
# =============================================================================

write_csv(
    CSV_DIR / "part97_part90_vs_part92.csv",
    [
        "Metric",
        "Part90",
        "Part92",
        "Delta_Part92_minus_Part90",
    ],
    CLASSIFICATION_COMPARISON_ROWS,
)


# =============================================================================
# SAVE PROJECT STATISTICS CSV
# =============================================================================

project_statistics = [
    ["Segmentation global foreground Dice", SEGMENTATION["global_foreground_dice"]],
    ["Segmentation median-case foreground Dice", SEGMENTATION["median_case_foreground_dice"]],
    ["Part90 accuracy", CLASSIFICATION_BASELINE["accuracy"]],
    ["Part90 majority baseline", CLASSIFICATION_BASELINE["majority_baseline"]],
    ["Part90 balanced accuracy", CLASSIFICATION_BASELINE["balanced_accuracy"]],
    ["Part90 Moderate recall", CLASSIFICATION_BASELINE["moderate_recall"]],
    ["Part90 Severe recall", CLASSIFICATION_BASELINE["severe_recall"]],
    ["Part92 accuracy", CLASSIFICATION_FINAL["accuracy"]],
    ["Part92 balanced accuracy", CLASSIFICATION_FINAL["balanced_accuracy"]],
    ["Part92 macro F1", CLASSIFICATION_FINAL["macro_f1"]],
    ["Part92 Moderate recall", CLASSIFICATION_FINAL["moderate_recall"]],
    ["Part92 Severe recall", CLASSIFICATION_FINAL["severe_recall"]],
    ["Integrated success rate", INTEGRATED["success_rate"]],
    ["Explainability success rate", EXPLAINABILITY["success_rate"]],
    ["Explainability visualizations", EXPLAINABILITY["visualizations"]],
]

write_csv(
    CSV_DIR / "part97_project_statistics.csv",
    ["Metric", "Value"],
    project_statistics,
)


# =============================================================================
# SAVE REPRODUCIBILITY CSV
# =============================================================================

repro_rows = []

for component, info in REPRODUCIBILITY.items():
    repro_rows.append(
        [
            component,
            info["status"],
            info["evidence"],
        ]
    )

write_csv(
    CSV_DIR / "part97_reproducibility_audit.csv",
    ["Component", "Status", "Evidence"],
    repro_rows,
)


# =============================================================================
# SAVE MODEL INVENTORY CSV
# =============================================================================

model_inventory = [
    [
        "Part84",
        "Swin-UNETR",
        SEGMENTATION["parameters"],
        SEGMENTATION["training_epochs"],
        SEGMENTATION["checkpoint"],
        "Final reproducible segmentation model",
    ],
    [
        "Part90",
        "Lightweight3DCNN",
        CLASSIFICATION_BASELINE["parameters"],
        5,
        "outputs/classification/rsna_part90_controlled_baseline/"
        "checkpoints/part90_best_model.pth",
        "Controlled baseline",
    ],
    [
        "Part92",
        "Part92CNN",
        CLASSIFICATION_FINAL["parameters"],
        CLASSIFICATION_FINAL["training_epochs"],
        "outputs/classification/rsna_part92_class_imbalance_aware_training/"
        "checkpoints/part92_best_model.pth",
        "Final imbalance-aware classifier",
    ],
]

write_csv(
    CSV_DIR / "part97_model_inventory.csv",
    [
        "Part",
        "Model",
        "Parameters",
        "Epochs",
        "Checkpoint",
        "Role",
    ],
    model_inventory,
)


# =============================================================================
# SAVE JSON SUMMARY
# =============================================================================

json_summary = {
    "project_title": PROJECT_TITLE,
    "generated_at": datetime.now().isoformat(),
    "final_status": "PASS — FINAL MSc REPORT PACKAGE GENERATED",
    "segmentation": SEGMENTATION,
    "classification_baseline_part90": CLASSIFICATION_BASELINE,
    "classification_final_part92": CLASSIFICATION_FINAL,
    "integrated_inference": INTEGRATED,
    "explainability": EXPLAINABILITY,
    "reproducibility": REPRODUCIBILITY,
    "historical_part71": {
        "reported_dice": 0.044026,
        "status": "NON-REPRODUCIBLE",
        "included_in_final_claims": False,
    },
    "important_methodological_caveat": (
        "Part90 and Part92 use different architectures, so their comparison "
        "is not a perfectly isolated single-variable class-weighting ablation."
    ),
    "clinical_status": (
        "Research prototype only; not clinically validated or suitable "
        "for clinical deployment."
    ),
}

json_path = REPORT_DIR / "part97_final_msc_report_summary.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_summary, f, indent=4)


# =============================================================================
# SAVE REPORT COPY IN reports/
# =============================================================================

report_copy = REPORT_DIR / "part97_final_msc_report.txt"
write_text(report_copy, complete_report)


# =============================================================================
# PRINT FINAL TERMINAL SUMMARY
# =============================================================================

print("=" * 90)
print("PART 97 — FINAL MSc PROJECT REPORT & RESULTS DISCUSSION PACKAGE")
print("=" * 90)

print(f"Project root : {ROOT}")
print(f"Output dir   : {OUTPUT_DIR}")
print()

print("FINAL VERIFIED RESULTS")
print("-" * 90)

print(
    f"Segmentation global foreground Dice : "
    f"{SEGMENTATION['global_foreground_dice']:.6f}"
)

print(
    f"Segmentation median-case Dice      : "
    f"{SEGMENTATION['median_case_foreground_dice']:.6f}"
)

print(
    f"Part90 baseline accuracy            : "
    f"{CLASSIFICATION_BASELINE['accuracy']:.6f}"
)

print(
    f"Part90 majority baseline            : "
    f"{CLASSIFICATION_BASELINE['majority_baseline']:.6f}"
)

print(
    f"Part92 final accuracy               : "
    f"{CLASSIFICATION_FINAL['accuracy']:.6f}"
)

print(
    f"Part92 balanced accuracy            : "
    f"{CLASSIFICATION_FINAL['balanced_accuracy']:.6f}"
)

print(
    f"Part92 macro F1                     : "
    f"{CLASSIFICATION_FINAL['macro_f1']:.6f}"
)

print(
    f"Part92 Moderate recall              : "
    f"{CLASSIFICATION_FINAL['moderate_recall']:.6f}"
)

print(
    f"Part92 Severe recall                : "
    f"{CLASSIFICATION_FINAL['severe_recall']:.6f}"
)

print(
    f"Integrated inference success       : "
    f"{INTEGRATED['success_rate']:.4f}"
)

print(
    f"Explainability success             : "
    f"{EXPLAINABILITY['success_rate']:.4f}"
)

print(
    f"Explainability visualizations      : "
    f"{EXPLAINABILITY['visualizations']}"
)

print()
print("REPRODUCIBILITY")
print("-" * 90)
print("Part84 final segmentation          : REPRODUCIBLE")
print("Part90 baseline classification     : REPRODUCIBLE")
print("Part92 final classification        : REPRODUCIBLE")
print("Part94 integrated inference        : VALIDATED")
print("Part95 explainability              : VALIDATED")
print("Historical Part71 Dice 0.044026    : NON-REPRODUCIBLE")
print()

print("IMPORTANT REPORTING RULES")
print("-" * 90)
print("1. Do not present 77.54% as strong classification accuracy.")
print("2. Report balanced accuracy and macro F1 for classification.")
print("3. Report global foreground Dice as the primary segmentation metric.")
print("4. Do not use historical Part71 Dice 0.044026 as a final result.")
print("5. State that Part90 and Part92 use different architectures.")
print("6. Describe Part95 as spatial/output-level explainability.")
print("7. Do not claim clinical validation or clinical deployment readiness.")
print()

print("GENERATED FILES")
print("-" * 90)
print(report_txt)
print(report_md)
print(report_copy)
print(json_path)
print(CSV_DIR / "part97_final_results_table.csv")
print(CSV_DIR / "part97_part90_vs_part92.csv")
print(CSV_DIR / "part97_project_statistics.csv")
print(CSV_DIR / "part97_reproducibility_audit.csv")
print(CSV_DIR / "part97_model_inventory.csv")

print()
print("=" * 90)
print("PART 97 FINAL RESULT")
print("=" * 90)
print("PASS — FINAL MSc REPORT PACKAGE GENERATED")
print("=" * 90)