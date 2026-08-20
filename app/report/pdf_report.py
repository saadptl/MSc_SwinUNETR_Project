
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


def _safe(value):
    return "N/A" if value is None else str(value)


def _image_for_report(image):
    if image is None:
        return None

    if isinstance(image, np.ndarray):
        from PIL import Image as PILImage
        from io import BytesIO

        pil = PILImage.fromarray(
            np.uint8(np.clip(image, 0, 255))
        )

        buffer = BytesIO()
        pil.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    return str(image)


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(
        18 * mm,
        height - 12 * mm,
        "Lumbar Spine AI — Research Analysis Report",
    )

    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(
        width - 18 * mm,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def build_pdf_report(
    result: dict,
    xai_result: dict | None,
    output_path: str | Path,
    xai_paths: dict | None = None,
    project_title: str = (
        "Automated Lumbar Spine Disease Detection "
        "and Classification from MRI Images"
    ),
):
    """
    Build a PDF from an already completed inference session.

    IMPORTANT:
    This module intentionally has no Streamlit dependency.
    Dashboard session-state values are passed through xai_paths.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xai_paths = xai_paths or {}

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=19,
        leading=23,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        spaceBefore=8,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="Lumbar Spine AI Analysis Report",
        author="MSc Computer Science — Data Science",
    )

    story = []

    prediction = result["prediction"]
    series = result["series"]
    selected = result["selected"]

    generated_at = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    # ------------------------------------------------------------------
    # COVER / EXECUTIVE SUMMARY
    # ------------------------------------------------------------------
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("LUMBAR SPINE AI", title_style))
    story.append(Paragraph(project_title, subtitle_style))

    summary_data = [
        ["Analysis Date", generated_at],
        ["Prediction", prediction["class_name"]],
        ["Confidence", f"{prediction['confidence']:.2f}%"],
        ["DICOM Slices", _safe(series["number_of_slices"])],
        ["Inference Device", _safe(prediction["device"])],
        ["Input Tensor", _safe(result["tensor_shape"])],
    ]

    table = Table(
        summary_data,
        colWidths=[45 * mm, 125 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("Purpose", heading_style))
    story.append(Paragraph(
        "This report documents one inference session from the "
        "deployment dashboard. It records the DICOM input, "
        "model input, prediction probabilities and, when available, "
        "the Grad-CAM explanation.",
        body_style,
    ))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # MRI INPUT
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "1. MRI INPUT AND SLICE SELECTION",
        heading_style,
    ))

    input_rows = [
        ["Channel", "Role", "Filename", "Instance"],
        [
            "0", "Previous",
            selected["channel_0_previous"]["filename"],
            _safe(selected["channel_0_previous"]["instance_number"]),
        ],
        [
            "1", "Middle",
            selected["channel_1_middle"]["filename"],
            _safe(selected["channel_1_middle"]["instance_number"]),
        ],
        [
            "2", "Next",
            selected["channel_2_next"]["filename"],
            _safe(selected["channel_2_next"]["instance_number"]),
        ],
    ]

    table = Table(
        input_rows,
        colWidths=[20 * mm, 28 * mm, 75 * mm, 35 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 8 * mm))

    images = result.get("processed_images")
    if images:
        image_flow = []

        for image in images:
            source = _image_for_report(image)
            if source:
                image_flow.append(
                    Image(
                        source,
                        width=52 * mm,
                        height=52 * mm,
                    )
                )

        if image_flow:
            while len(image_flow) < 3:
                image_flow.append(Spacer(52 * mm, 52 * mm))

            image_table = Table(
                [image_flow[:3]],
                colWidths=[56 * mm, 56 * mm, 56 * mm],
            )
            image_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(image_table)

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "The three displayed slices correspond to the three channels "
        "supplied to the classifier.",
        small_style,
    ))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "2. MODEL PREDICTION",
        heading_style,
    ))

    story.append(Paragraph(
        f"<b>Predicted class:</b> {prediction['class_name']}<br/>"
        f"<b>Confidence:</b> {prediction['confidence']:.2f}%",
        body_style,
    ))

    probability_rows = [["Class", "Probability"]]
    for name, probability in prediction["probabilities"].items():
        probability_rows.append([
            name,
            f"{probability:.4f}%",
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[80 * mm, 50 * mm],
        repeatRows=1,
    )
    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(probability_table)
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("Model Input", heading_style))
    story.append(Paragraph(
        f"Tensor shape: <b>{_safe(result['tensor_shape'])}</b><br/>"
        "Input construction: Previous / Middle / Next slices<br/>"
        "Preprocessing: Min-Max normalization → CLAHE → "
        "224 × 224 resize → /255",
        body_style,
    ))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # XAI
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "3. EXPLAINABLE AI — GRAD-CAM",
        heading_style,
    ))

    if xai_result:
        story.append(Paragraph(
            f"<b>Target class:</b> "
            f"{_safe(xai_result.get('target_class_name'))}<br/>"
            f"<b>Feature representation:</b> "
            f"{_safe(xai_result.get('feature_shape'))}<br/>"
            f"<b>CAM mean:</b> "
            f"{float(xai_result.get('cam_mean', 0)):.6f}<br/>"
            f"<b>CAM standard deviation:</b> "
            f"{float(xai_result.get('cam_std', 0)):.6f}<br/>"
            f"<b>Maximum logit consistency error:</b> "
            f"{float(xai_result.get('max_logit_error', 0)):.2e}",
            body_style,
        ))

        xai_images = []

        for key in ("heatmap", "overlay"):
            path = xai_paths.get(key)

            if path and Path(path).exists():
                xai_images.append(
                    Image(
                        str(path),
                        width=76 * mm,
                        height=58 * mm,
                    )
                )

        if xai_images:
            while len(xai_images) < 2:
                xai_images.append(Spacer(76 * mm, 58 * mm))

            story.append(Table(
                [xai_images[:2]],
                colWidths=[82 * mm, 82 * mm],
                style=TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]),
            ))

        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph(
            "Interpretation: Grad-CAM visualizes spatial feature regions "
            "associated with the selected model prediction. It is an "
            "explainability aid and is not a segmentation mask or "
            "definitive anatomical disease localization.",
            body_style,
        ))
    else:
        story.append(Paragraph(
            "Grad-CAM was not generated for this analysis session. "
            "The report therefore contains the model prediction without "
            "an XAI visualization.",
            body_style,
        ))

    story.append(PageBreak())

    # ------------------------------------------------------------------
    # TECHNICAL SUMMARY
    # ------------------------------------------------------------------
    story.append(Paragraph(
        "4. TECHNICAL SUMMARY",
        heading_style,
    ))

    technical_rows = [
        ["Item", "Value"],
        ["Inference device", _safe(prediction["device"])],
        ["Tensor shape", _safe(result["tensor_shape"])],
        ["DICOM slice count", _safe(series["number_of_slices"])],
        ["Channel 0", selected["channel_0_previous"]["filename"]],
        ["Channel 1", selected["channel_1_middle"]["filename"]],
        ["Channel 2", selected["channel_2_next"]["filename"]],
        ["Grad-CAM", "Available" if xai_result else "Not generated"],
    ]

    table = Table(
        technical_rows,
        colWidths=[55 * mm, 105 * mm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph(
        "5. RESEARCH LIMITATIONS AND DISCLAIMER",
        heading_style,
    ))

    story.append(Paragraph(
        "This system is an academic research and demonstration "
        "prototype. Model probabilities and Grad-CAM visualizations "
        "represent the behavior of the trained machine-learning system "
        "for the supplied input. They must not be treated as a clinical "
        "diagnosis, definitive disease localization, or a replacement "
        "for qualified medical interpretation.",
        body_style,
    ))

    story.append(Paragraph(
        "This report documents an inference session and does not modify "
        "model weights, retrain the network, or alter the original "
        "checkpoint.",
        body_style,
    ))

    doc.build(
        story,
        onFirstPage=_header_footer,
        onLaterPages=_header_footer,
    )

    return str(output_path)
