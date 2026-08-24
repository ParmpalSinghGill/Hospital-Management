"""Bare clock times, 'after X' preferences, and load-balanced doctor choice."""
from __future__ import annotations

from datetime import datetime

import db_appointments
import db_time
from tests.test_db_appointments import _book


def test_bare_clock_means_next_occurrence():
    morning = datetime(2026, 8, 24, 9, 30)
    evening = datetime(2026, 8, 24, 16, 0)
    assert db_time.normalize_appointment_time("12:00 PM", now=morning) == "2026-08-24 12:00"
    assert db_time.normalize_appointment_time("12:00 PM", now=evening) == "2026-08-25 12:00"
    assert db_time.normalize_appointment_time("3 pm", now=morning) == "2026-08-24 15:00"


def test_after_prefix_parses_and_flags():
    now = datetime(2026, 8, 24, 9, 30)
    for raw in ("after 12pm", "after 12:00 PM", "any time after 12 pm", "later than 12 pm"):
        assert db_time.is_after_preference(raw), raw
        assert db_time.normalize_appointment_time(raw, now=now) == "2026-08-24 12:00", raw
    assert not db_time.is_after_preference("12:00 PM")
    assert db_time.normalize_appointment_time("after tomorrow 10 am", now=now) == "2026-08-25 10:00"


def test_after_preference_never_offers_earlier_slots(isolated_db):
    _book("Pat One", "9000000001", "DOC-0001", "2026-12-01 12:00")
    nearest = db_appointments.find_nearest_available_times(
        "DOC-0001", "after 2026-12-01 12:00", limit=5
    )
    assert nearest, "expected forward slots"
    assert all(t >= "2026-12-01 12:00" for t in nearest), nearest
    # plain exact request may still offer earlier same-day slots (nearest-around)
    plain = db_appointments.find_nearest_available_times("DOC-0001", "2026-12-01 12:00", limit=5)
    assert any(t < "2026-12-01 12:00" for t in plain)


def test_ranking_prefers_less_loaded_doctor(isolated_db):
    import db_doctors
    docs = db_doctors._load_doctors()[:2]
    assert len(docs) == 2
    busy, free = docs[0], docs[1]
    _book("Pat A", "9000000002", busy["doctor_id"], "2026-12-02 09:00")
    _book("Pat B", "9000000003", busy["doctor_id"], "2026-12-02 09:30")
    ranked = db_appointments.rank_doctors_for_preferred_time(docs, "2026-12-02 11:00")
    assert ranked[0]["doctor_id"] == free["doctor_id"]
    assert ranked[0]["day_appointment_count"] == 0
    assert ranked[1]["day_appointment_count"] == 2


def test_before_window_only_offers_earlier_slots(isolated_db):
    nearest = db_appointments.find_nearest_available_times(
        "DOC-0001", "2026-12-01 14:00", limit=10, time_window="before"
    )
    assert nearest, "expected earlier slots"
    assert all(t < "2026-12-01 14:00" for t in nearest), nearest


def test_explicit_after_window_without_wording(isolated_db):
    nearest = db_appointments.find_nearest_available_times(
        "DOC-0001", "2026-12-01 14:00", limit=10, time_window="after"
    )
    assert nearest
    assert all(t >= "2026-12-01 14:00" for t in nearest), nearest


def test_list_doctors_time_window_parameter(isolated_db):
    import json

    from Tools import list_doctors

    out = json.loads(
        list_doctors.invoke(
            {"department": "Cardiology", "preferred_time": "2026-12-01 2:00 PM", "time_window": "after"}
        )
    )
    assert out["ok"] and out["time_window"] == "after"
    doc = out["doctors"][0]
    assert doc["free_at_preferred_time"] is True
    assert doc["nearest_times"] and all(t >= "2026-12-01 14:00" for t in doc["nearest_times"])

    before = json.loads(
        list_doctors.invoke(
            {"department": "Cardiology", "preferred_time": "before 2026-12-01 2:00 PM"}
        )
    )
    assert before["time_window"] == "before"  # inferred from wording
    bdoc = before["doctors"][0]
    assert all(t < "2026-12-01 14:00" for t in bdoc.get("nearest_times") or []), bdoc
