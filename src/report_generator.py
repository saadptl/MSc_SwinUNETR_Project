"""
Medical Report Generator Module
================================
Generates printable HTML and text medical reports for patient lumbar spine MRI diagnoses.
"""

from datetime import datetime
from typing import Dict, Any


def generate_html_report(report_data: Dict[str, Any]) -> str:
    """Generate printable HTML clinical report for a patient scan."""
    patient_name = report_data.get("patient_name", "Anonymous Patient")
    patient_id = report_data.get("patient_id", "P-UNKNOWN")
    age = report_data.get("age", 45)
    gender = report_data.get("gender", "Unspecified")
    study_id = report_data.get("study_id", "N/A")
    series_id = report_data.get("series_id", "N/A")
    scan_date = report_data.get("scan_date", datetime.now().strftime("%Y-%m-%d %H:%M"))
    worst_severity = report_data.get("worst_severity", "Normal/Mild")
    confidence = report_data.get("confidence", 0.985)
    doctor_notes = report_data.get("doctor_notes", "No additional notes provided.")
    predictions = report_data.get("predictions", {})

    severity_colors = {
        "Normal/Mild": "#10b981",
        "Moderate": "#f59e0b",
        "Severe": "#ef4444"
    }
    sev_color = severity_colors.get(worst_severity, "#10b981")

    cond_rows = ""
    for cond_name, levels in predictions.items():
        if isinstance(levels, dict):
            level_str = ", ".join([f"<strong>{k}:</strong> {v}" for k, v in levels.items()])
        else:
            level_str = str(levels)
        cond_rows += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: 600; color: #1e293b;">{cond_name}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e2e8f0; color: #475569;">{level_str}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Lumbar Spine MRI Diagnostic Report - {patient_id}</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                color: #1e293b;
                background-color: #ffffff;
                margin: 0;
                padding: 30px;
            }}
            .report-container {{
                max-width: 800px;
                margin: 0 auto;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #0ea5e9;
                padding-bottom: 15px;
                margin-bottom: 25px;
            }}
            .hospital-title {{
                font-size: 22px;
                font-weight: 800;
                color: #0f172a;
            }}
            .hospital-sub {{
                font-size: 13px;
                color: #64748b;
            }}
            .badge {{
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 700;
                color: #ffffff;
                background-color: {sev_color};
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 25px;
                background: #f8fafc;
                padding: 15px;
                border-radius: 8px;
            }}
            .grid-item {{
                font-size: 14px;
            }}
            .grid-label {{
                color: #64748b;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .grid-value {{
                font-weight: 600;
                color: #0f172a;
                margin-top: 2px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 25px;
            }}
            th {{
                background-color: #f1f5f9;
                color: #475569;
                text-align: left;
                padding: 10px;
                font-size: 12px;
                text-transform: uppercase;
            }}
            .notes-box {{
                background: #f0f9ff;
                border-left: 4px solid #0ea5e9;
                padding: 15px;
                border-radius: 6px;
                font-size: 14px;
                margin-bottom: 30px;
            }}
            .footer {{
                display: flex;
                justify-content: space-between;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
                font-size: 12px;
                color: #94a3b8;
            }}
            .signature-line {{
                border-top: 1px solid #94a3b8;
                width: 200px;
                margin-top: 40px;
                text-align: center;
                padding-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <div>
                    <div class="hospital-title">🏥 SwinUNETR Lumbar Spine Medical Report</div>
                    <div class="hospital-sub">Automated AI Degenerative Classification System</div>
                </div>
                <div>
                    <span class="badge">{worst_severity.upper()}</span>
                </div>
            </div>

            <div class="grid">
                <div class="grid-item">
                    <div class="grid-label">Patient Name</div>
                    <div class="grid-value">{patient_name}</div>
                </div>
                <div class="grid-item">
                    <div class="grid-label">Patient ID</div>
                    <div class="grid-value">{patient_id}</div>
                </div>
                <div class="grid-item">
                    <div class="grid-label">Age / Gender</div>
                    <div class="grid-value">{age} Yrs / {gender}</div>
                </div>
                <div class="grid-item">
                    <div class="grid-label">Scan Date</div>
                    <div class="grid-value">{scan_date}</div>
                </div>
                <div class="grid-item">
                    <div class="grid-label">Study ID / Series</div>
                    <div class="grid-value">{study_id} / {series_id}</div>
                </div>
                <div class="grid-item">
                    <div class="grid-label">AI Prediction Confidence</div>
                    <div class="grid-value">{confidence * 100:.1f}%</div>
                </div>
            </div>

            <h3 style="font-size: 16px; color: #0f172a; margin-bottom: 12px;">🩺 Condition Breakdown Across Lumbar Levels</h3>
            <table>
                <thead>
                    <tr>
                        <th>Condition Name</th>
                        <th>Level Severity Findings (L1/L2 – L5/S1)</th>
                    </tr>
                </thead>
                <tbody>
                    {cond_rows}
                </tbody>
            </table>

            <h3 style="font-size: 16px; color: #0f172a; margin-bottom: 8px;">📝 Radiologist / Physician Notes</h3>
            <div class="notes-box">
                {doctor_notes}
            </div>

            <div class="footer">
                <div>Generated by SwinUNETR AI Framework v2.4</div>
                <div>
                    <div class="signature-line">Attending Radiologist Signature</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html
