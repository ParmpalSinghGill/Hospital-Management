"""Unit tests for appointment time parsing and clinic hours."""

from __future__ import annotations

from datetime import datetime

import db_time


def test_normalize_iso_and_space_formats():
    assert db_time.normalize_appointment_time("2026-07-22 10:00") == "2026-07-22 10:00"
    assert db_time.normalize_appointment_time("2026-07-22T10:30") == "2026-07-22 10:30"


def test_normalize_relative_day_with_clock():
    now = datetime(2026, 7, 21, 8, 0, 0)
    assert db_time.normalize_appointment_time("tomorrow 10 am", now=now) == "2026-07-22 10:00"
    assert db_time.normalize_appointment_time("today 3 pm", now=now) == "2026-07-21 15:00"


def test_normalize_day_only_returns_none():
    now = datetime(2026, 7, 21, 8, 0, 0)
    assert db_time.normalize_appointment_time("today", now=now) is None
    assert db_time.resolve_appointment_day("today", now=now).isoformat() == "2026-07-21"


def test_parse_availability_anchor_day_only():
    now = datetime(2026, 7, 21, 8, 0, 0)
    anchor, day_only = db_time.parse_availability_anchor("today", now=now)
    assert day_only is True
    assert anchor is not None
    assert anchor.date().isoformat() == "2026-07-21"


def test_clinic_hours():
    assert db_time.is_within_clinic_hours("2026-07-22 09:00")
    assert db_time.is_within_clinic_hours("2026-07-22 16:50")
    assert not db_time.is_within_clinic_hours("2026-07-22 17:00")
    assert not db_time.is_within_clinic_hours("2026-07-22 08:50")
    assert not db_time.is_within_clinic_hours("not-a-time")


def test_classify_appointment_timing_buckets():
    now = datetime(2026, 7, 21, 12, 0, 0)
    past = db_time.classify_appointment_timing("2026-07-21 10:00", now=now)
    soon = db_time.classify_appointment_timing("2026-07-21 14:00", now=now)
    later = db_time.classify_appointment_timing("2026-07-25 10:00", now=now)
    assert past["timing_bucket"] == "past"
    assert soon["timing_bucket"] == "soon"
    assert later["timing_bucket"] == "later"


def test_iter_clinic_slots_ten_minute():
    day = datetime(2026, 7, 22).date()
    slots = db_time._iter_clinic_slots_on_day(day, slot_minutes=10)
    assert slots[0].strftime("%H:%M") == "09:00"
    assert slots[-1].strftime("%H:%M") == "16:50"
    assert len(slots) == 48
