"""Unit tests for appointments, conflicts, and availability."""

from __future__ import annotations

import db_appointments
import db_doctors
import db_patients
import pytest


def _book(patient_name: str, phone: str, doctor_id: str, when: str) -> dict:
    patient = db_patients._get_or_create_patient(patient_name, phone)
    doctor = db_doctors._get_doctor_by_id(doctor_id)
    appt = {
        "appointment_id": db_appointments._get_next_appointment_id(),
        "patient_id": patient["patient_id"],
        "doctor_id": doctor["doctor_id"],
        "doctor": doctor["name"],
        "department": doctor["department"],
        "time": when,
        "status": "BOOKED",
    }
    db_appointments._insert_appointment(appt)
    return appt


def test_check_slot_bookable_outside_hours(isolated_db):
    refusal = db_appointments.check_slot_bookable("DOC-0001", "2026-12-01 18:00")
    assert refusal and refusal["reason"] == "outside_clinic_hours"


def test_check_slot_bookable_lunch(isolated_db):
    refusal = db_appointments.check_slot_bookable("DOC-0001", "2026-12-01 14:20")
    assert refusal and refusal["reason"] == "doctor_unavailable"


def test_insert_conflict_and_nearest(isolated_db):
    when = "2026-12-01 10:00"
    first = _book("Pat One", "9000000001", "DOC-0001", when)
    refusal = db_appointments.check_slot_bookable(
        "DOC-0001",
        when,
        doctor_name="Dr. Test Cardio",
    )
    assert refusal and refusal["reason"] == "time_conflict"
    nearest = db_appointments.find_nearest_available_times("DOC-0001", when, limit=3)
    assert when not in nearest
    assert len(nearest) >= 1
    assert first["appointment_id"]


def test_find_available_doctors_at_time(isolated_db):
    when = "2026-12-01 10:00"
    _book("Pat One", "9000000001", "DOC-0001", when)
    free = db_appointments.find_available_doctors_at_time(
        when,
        department="Cardiology",
        exclude_doctor_id="DOC-0001",
    )
    # Only one cardiologist in seed; after exclude, none in Cardiology
    assert all(d["doctor_id"] != "DOC-0001" for d in free)
    free_any = db_appointments.find_available_doctors_at_time(when)
    assert any(d["doctor_id"] == "DOC-0002" for d in free_any)


def test_active_appointment_and_complete_past(isolated_db):
    past = _book("Pat Past", "9000000002", "DOC-0001", "2020-01-01 10:00")
    assert db_appointments._find_active_appointment_for_patient(past["patient_id"]) is None
    completed = db_appointments._complete_past_appointments_for_patient(past["patient_id"])
    assert past["appointment_id"] in completed
    rows = db_appointments._load_all_appointments()
    row = next(a for a in rows if a["appointment_id"] == past["appointment_id"])
    assert row["status"] == "COMPLETED"


def test_enrich_appointment(isolated_db):
    appt = _book("Pat Enrich", "9000000003", "DOC-0002", "2026-12-03 11:00")
    enriched = db_appointments._enrich_appointment(appt)
    assert enriched["patient_name"] == "Pat Enrich"
    assert enriched["doctor"] == "Dr. Test Dental"
    assert enriched["department"] == "Dentistry"


def test_department_booking_guidance(isolated_db):
    appt = _book("Pat Guide", "9000000004", "DOC-0001", "2026-12-10 10:00")
    enriched = db_appointments._enrich_appointment(appt)
    guidance = db_appointments.build_department_booking_guidance(enriched, "Cardiology")
    assert guidance["action"] in ("offer_prepone", "inform_existing_soon")
    none = db_appointments.build_department_booking_guidance(None, "Dentistry")
    assert none["action"] == "offer_new_booking"


def test_cancel_via_update(isolated_db):
    appt = _book("Pat Cancel", "9000000005", "DOC-0001", "2026-12-04 10:00")
    appt["status"] = "CANCELLED"
    db_appointments._update_appointment(appt)
    assert db_appointments._find_active_appointment_for_patient(appt["patient_id"]) is None
