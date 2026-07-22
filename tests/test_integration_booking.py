"""Integration: multi-step appointment lifecycle via Tools."""

from __future__ import annotations

import json

from Tools import (
    book_appointment,
    cancel_appointment,
    list_doctors,
    lookup_patient,
    reschedule_appointment,
    save_patient,
)


def _j(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


def test_full_patient_journey_book_reschedule_cancel(isolated_db):
    """Identity → doctors → book → conflict → reschedule → cancel."""
    phone = "9811111111"
    name = "Integration Patient"

    # New patient
    looked = _j(lookup_patient, phone=phone)
    assert looked["ok"] and looked.get("is_new")

    saved = _j(save_patient, patient_name=name, phone=phone)
    assert saved["ok"] and saved["created"]

    verified = _j(lookup_patient, phone=phone, patient_name=name, department="Cardiology")
    assert verified["verified"] is True
    assert verified["booking_guidance"]["action"] == "offer_new_booking"

    docs = _j(list_doctors, department="Cardiology", preferred_time="2026-12-15 10:00")
    assert docs["ok"] and docs["count"] >= 1
    doctor_name = docs["doctors"][0]["name"]

    booked = _j(
        book_appointment,
        patient_name=name,
        phone=phone,
        doctor=doctor_name,
        time="2026-12-15 10:00",
    )
    assert booked["ok"] is True, booked
    aid = booked["appointment"]["appointment_id"]

    # Same slot conflict for another patient
    conflict = _j(
        book_appointment,
        patient_name="Other Person",
        phone="9822222222",
        doctor=doctor_name,
        time="2026-12-15 10:00",
    )
    assert conflict["ok"] is False
    assert conflict.get("nearest_times")

    # Active appointment blocks second booking for same patient
    blocked = _j(
        book_appointment,
        patient_name=name,
        phone=phone,
        doctor="Dr. Test Dental",
        time="2026-12-15 11:00",
    )
    assert blocked["ok"] is False

    moved = _j(reschedule_appointment, appointment_id=aid, new_time="2026-12-15 11:30")
    assert moved["ok"] is True
    assert moved["appointment"]["time"] == "2026-12-15 11:30"

    # Lookup sees active appointment
    again = _j(lookup_patient, phone=phone, patient_name=name)
    assert again["active_appointment"]["appointment_id"] == aid

    cancelled = _j(cancel_appointment, phone=phone)
    assert cancelled["ok"] is True
    assert cancelled["appointment"]["status"] == "CANCELLED"

    final = _j(lookup_patient, phone=phone, patient_name=name)
    assert final["active_appointment"] is None


def test_lunch_slot_refused_with_nearest(isolated_db):
    saved = _j(save_patient, patient_name="Lunch Pat", phone="9833333333")
    assert saved["ok"]
    out = _j(
        book_appointment,
        patient_name="Lunch Pat",
        phone="9833333333",
        doctor="Dr. Test Cardio",
        time="2026-12-15 14:20",
    )
    assert out["ok"] is False
    assert out.get("reason") == "doctor_unavailable"
    assert out.get("nearest_times")
