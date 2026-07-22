"""Tests for prescription helpers and database clear."""

from __future__ import annotations

import db_core
import db_doctors
import db_patients
import db_prescriptions


def test_prescription_roundtrip(isolated_db):
    patient = db_patients._get_or_create_patient("Rx Patient", "9121212121")
    doctor = db_doctors._get_doctor_by_id("DOC-0001")
    rx_id = db_prescriptions._get_next_prescription_id()
    db_prescriptions._save_prescriptions(
        [
            {
                "prescription_id": rx_id,
                "patient_id": patient["patient_id"],
                "doctor_id": doctor["doctor_id"],
                "medicine_name": "MediClear",
                "timing": "Once daily after breakfast",
            }
        ]
    )
    found = db_prescriptions._find_prescriptions(
        patient_id=patient["patient_id"],
    )
    assert found["ok"] is True
    assert found["count"] == 1
    assert found["prescriptions"][0]["medicine_name"] == "MediClear"
    assert found["prescriptions"][0].get("doctor_name") == "Dr. Test Cardio"


def test_clear_table(isolated_db):
    before = len(db_doctors._load_doctors())
    assert before == 3
    removed = db_core.clear_table("doctors")
    assert removed == 3
    assert db_doctors._load_doctors() == []
