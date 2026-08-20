"""
Patient Database Storage Module
================================
SQLite storage engine for saving MRI upload scans, patient metadata,
SwinUNETR diagnosis predictions, doctor notes, and report history.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = r"C:\Saad\Msc Major Project Swin Unetr Framework\MSc_SwinUNETR_Project\outputs\patient_database.db"


def get_db_connection():
    """Establish connection to SQLite database and ensure schema exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    """Create database tables if they do not exist."""
    cursor = conn.cursor()
    
    # Patient Records table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        patient_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        created_at TEXT NOT NULL
    )
    """)
    
    # Diagnostic Reports / MRI Scans table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_reports (
        report_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL,
        study_id TEXT NOT NULL,
        series_id TEXT,
        scan_date TEXT NOT NULL,
        scan_type TEXT DEFAULT 'Lumbar Spine MRI',
        file_name TEXT NOT NULL,
        image_path TEXT,
        worst_severity TEXT NOT NULL,
        confidence REAL NOT NULL,
        predictions_json TEXT NOT NULL,
        doctor_notes TEXT,
        status TEXT DEFAULT 'Completed',
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    """)
    
    conn.commit()


def save_scan_report(
    patient_id: str,
    patient_name: str,
    age: int,
    gender: str,
    study_id: str,
    series_id: str,
    file_name: str,
    image_path: str,
    worst_severity: str,
    confidence: float,
    predictions: Dict[str, Any],
    doctor_notes: str = ""
) -> str:
    """Save or update a patient scan report record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Insert or ignore patient
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO patients (patient_id, patient_name, age, gender, created_at)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(patient_id) DO UPDATE SET
        patient_name=excluded.patient_name,
        age=excluded.age,
        gender=excluded.gender
    """, (patient_id, patient_name, age, gender, now_str))
    
    # 2. Insert scan report
    report_id = f"REP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{study_id[:6]}"
    predictions_json = json.dumps(predictions)
    
    cursor.execute("""
    INSERT INTO scan_reports (
        report_id, patient_id, study_id, series_id, scan_date,
        scan_type, file_name, image_path, worst_severity,
        confidence, predictions_json, doctor_notes, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id, patient_id, study_id, series_id, now_str,
        'Lumbar Spine MRI', file_name, image_path, worst_severity,
        confidence, predictions_json, doctor_notes, 'Completed'
    ))
    
    conn.commit()
    conn.close()
    return report_id


def get_all_reports() -> List[Dict[str, Any]]:
    """Retrieve all patient scan reports joined with patient demographic info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT r.*, p.patient_name, p.age, p.gender
    FROM scan_reports r
    JOIN patients p ON r.patient_id = p.patient_id
    ORDER BY r.scan_date DESC
    """)
    
    rows = cursor.fetchall()
    reports = []
    for r in rows:
        item = dict(r)
        if item.get("predictions_json"):
            try:
                item["predictions"] = json.loads(item["predictions_json"])
            except Exception:
                item["predictions"] = {}
        reports.append(item)
        
    conn.close()
    return reports


def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
    """Get a single report by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT r.*, p.patient_name, p.age, p.gender
    FROM scan_reports r
    JOIN patients p ON r.patient_id = p.patient_id
    WHERE r.report_id = ?
    """, (report_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        res = dict(row)
        if res.get("predictions_json"):
            try:
                res["predictions"] = json.loads(res["predictions_json"])
            except Exception:
                res["predictions"] = {}
        return res
    return None


def update_doctor_notes(report_id: str, doctor_notes: str):
    """Update doctor notes for a saved scan report."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE scan_reports SET doctor_notes = ? WHERE report_id = ?
    """, (doctor_notes, report_id))
    conn.commit()
    conn.close()


def delete_report(report_id: str):
    """Delete a scan report record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_reports WHERE report_id = ?", (report_id,))
    conn.commit()
    conn.close()
