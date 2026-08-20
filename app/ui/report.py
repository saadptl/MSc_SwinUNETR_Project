from pathlib import Path
from datetime import datetime

import streamlit as st

from ui.common import navigate, filename, safe, pct


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _session():
    s = st.session_state.get("analysis_session")
    return s if s and s.has_prediction else None


def _build_pdf(s):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.units import inch

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"lumbar_spine_ai_report_{stamp}.pdf"

    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=12)
    subtitle = ParagraphStyle("Sub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, textColor=colors.grey)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=9.5, leading=14)

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = [
        Paragraph("Lumbar Spine AI", title),
        Paragraph("Automated Lumbar Spine MRI Classification — Research Report", subtitle),
        Spacer(1, 18),
        Paragraph("1. Analysis Summary", h2),
    ]

    summary = [
        ["Prediction", safe(s.predicted_class)],
        ["Confidence", f"{pct(s.confidence):.2f}%"],
        ["Model", safe(s.model_name, "Swin Transformer")],
        ["Runtime", safe(s.device)],
        ["Input Tensor", safe(s.tensor_shape)],
        ["Generated", datetime.now().strftime("%d %B %Y, %H:%M")],
    ]
    table = Table(summary, colWidths=[1.55 * inch, 4.95 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E8ECF7")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story += [table, Spacer(1, 12)]

    story += [Paragraph("2. Classification Probabilities", h2)]
    probs = s.probabilities or {}
    ptable = Table([[name, f"{pct(probs.get(name, 0)):.2f}%"] for name in ["Normal/Mild", "Moderate", "Severe"]], colWidths=[4.8*inch, 1.7*inch])
    ptable.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 9), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold")]))
    story += [ptable, Spacer(1, 12)]

    story += [Paragraph("3. MRI Information", h2)]
    meta = s.series_metadata if isinstance(s.series_metadata, dict) else {}
    mtable = Table([
        ["Study ID", safe(s.study_id, "Not available from upload")],
        ["Series ID", safe(s.series_id, "Not available from upload")],
        ["DICOM Slices", str(meta.get("number_of_slices", "N/A"))],
        ["First Slice", filename(meta.get("first_file"))],
        ["Last Slice", filename(meta.get("last_file"))],
    ], colWidths=[1.55*inch, 4.95*inch])
    mtable.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9)]))
    story += [mtable, Spacer(1, 12)]

    story += [Paragraph("4. Model Input Construction", h2), Paragraph("Three adjacent DICOM slices are used as Channel 0 (Previous), Channel 1 (Middle), and Channel 2 (Next).", body), Spacer(1, 6)]
    files = s.selected_files or []
    labels = ["Channel 0 — Previous", "Channel 1 — Middle", "Channel 2 — Next"]
    itable = Table([[label, filename(path)] for label, path in zip(labels, files)], colWidths=[3.0*inch, 3.5*inch])
    itable.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9)]))
    story += [itable, Spacer(1, 12)]

    if s.has_xai:
        story += [Paragraph("5. Explainable AI", h2), Paragraph("Grad-CAM is included as a model-attention visualization. It is not a clinically validated disease localization or segmentation method.", body), Spacer(1, 8)]
        panel = s.xai_paths.get("panel")
        if panel and Path(panel).exists():
            img = Image(panel, width=6.5*inch, height=6.5*inch/3.0)
            story += [img, Spacer(1, 8)]

    story += [Paragraph("Research Disclaimer", h2), Paragraph("This report is generated for academic research, software demonstration, and project evaluation. It is not intended for clinical diagnosis or treatment decisions.", body)]

    doc.build(story)
    return path


def render_report():
    s = _session()
    st.title("📄 Research Report")
    st.caption("Professional PDF report generated from the current inference, XAI, and MRI session.")

    if s is None:
        st.info("Complete an MRI analysis before generating a report.")
        if st.button("🩻 Go to MRI Analysis", type="primary", use_container_width=True):
            navigate("🩻 MRI Analysis")
        return

    st.success("Analysis data is ready for report generation.")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Prediction", safe(s.predicted_class))
    with c2: st.metric("Confidence", f"{pct(s.confidence):.2f}%")
    with c3: st.metric("XAI", "Available" if s.has_xai else "Pending")

    if st.button("📄 Generate Professional PDF Report", type="primary", use_container_width=True, key="generate_pdf"):
        with st.spinner("Building the PDF report..."):
            try:
                path = _build_pdf(s)
                s.report_path = str(path)
                s.report_generated = True
                s.report_summary = {"prediction": s.predicted_class, "confidence": pct(s.confidence)}
                st.success("Professional PDF report generated successfully.")
            except Exception as exc:
                st.error("PDF report generation failed.")
                st.exception(exc)

    if s.has_report:
        path = Path(s.report_path)
        if path.exists():
            st.markdown("### 📥 Generated Report")
            st.download_button(
                "⬇️ Download Professional PDF Report",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_report",
            )
            st.code(str(path), language="text")

    st.markdown("---")
    st.markdown("### Workflow")
    st.write("MRI Analysis → Inference → Results → Grad-CAM (optional) → Professional PDF Report")
