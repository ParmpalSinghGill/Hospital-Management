"""Unit tests for LangChain tools (JSON contract)."""

from __future__ import annotations

import json

from Tools import (
    book_appointment,
    cancel_appointment,
    get_prescriptions,
    list_doctors,
    lookup_patient,
    reschedule_appointment,
    save_patient,
)


def _j(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


def test_lookup_asks_for_phone():
    out = _j(lookup_patient)
    assert out["ok"] is False
    assert out.get("needs_phone") is True


def test_save_and_lookup_returning(isolated_db):
    saved = _j(save_patient, patient_name="Eve Tester", phone="9444444444")
    assert saved["ok"] is True
    assert saved["created"] is True

    first = _j(lookup_patient, phone="9444444444")
    assert first["ok"] is True
    assert first["is_returning"] is True
    assert first.get("verified") is False
    assert first.get("confirm_name_from_db") == "Eve Tester"

    verified = _j(
        lookup_patient,
        phone="9444444444",
        patient_name="Eve Tester",
        department="Cardiology",
    )
    assert verified["verified"] is True
    assert "booking_guidance" in verified


def test_save_rejects_incomplete_phone(isolated_db):
    out = _j(save_patient, patient_name="X", phone="12345")
    assert out["ok"] is False
    assert out.get("incomplete_phone") is True


def test_list_doctors_department_and_preferred_time(isolated_db):
    out = _j(list_doctors, department="Cardiology", limit=5)
    assert out["ok"] is True
    assert out["count"] >= 1
    assert all("Cardio" in (d.get("department") or "") for d in out["doctors"])

    timed = _j(
        list_doctors,
        department="Cardiology",
        preferred_time="2026-12-01 10:00",
        limit=5,
    )
    assert timed["ok"] is True
    assert timed["count"] >= 1


def test_book_cancel_reschedule_flow(isolated_db):
    booked = _j(
        book_appointment,
        patient_name="Flow User",
        phone="9555555555",
        doctor="Dr. Test Cardio",
        time="2026-12-05 10:00",
    )
    assert booked["ok"] is True, booked
    aid = booked["appointment"]["appointment_id"]

    # second booking blocked while first is active
    blocked = _j(
        book_appointment,
        patient_name="Flow User",
        phone="9555555555",
        doctor="Dr. Test Dental",
        time="2026-12-05 11:00",
    )
    assert blocked["ok"] is False

    moved = _j(reschedule_appointment, appointment_id=aid, new_time="2026-12-05 11:00")
    assert moved["ok"] is True, moved
    assert moved["appointment"]["time"] == "2026-12-05 11:00"

    cancelled = _j(cancel_appointment, appointment_id=aid)
    assert cancelled["ok"] is True
    assert cancelled["appointment"]["status"] == "CANCELLED"


def test_book_conflict_returns_suggestions(isolated_db):
    first = _j(
        book_appointment,
        patient_name="First",
        phone="9666666666",
        doctor="Dr. Test Cardio",
        time="2026-12-06 10:00",
    )
    assert first["ok"] is True
    conflict = _j(
        book_appointment,
        patient_name="Second",
        phone="9777777777",
        doctor="Dr. Test Cardio",
        time="2026-12-06 10:00",
    )
    assert conflict["ok"] is False
    assert conflict.get("reason") == "time_conflict"
    assert conflict.get("nearest_times")


def test_get_prescriptions_needs_identity(isolated_db):
    out = _j(get_prescriptions)
    assert out["ok"] is False
    missing = _j(get_prescriptions, phone="9888888888", patient_name="Nobody")
    assert missing["ok"] is False
