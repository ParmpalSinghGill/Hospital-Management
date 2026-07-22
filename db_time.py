"""Appointment time parsing, clinic hours, and timing classification."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional, Tuple

from db_core import (
    APPOINTMENT_SOON_HOURS,
    _CLINIC_END,
    _CLINIC_START,
    _DEFAULT_SLOT_MINUTES,
)

_APPT_TIME_FMTS = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


_CLOCK_RE = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE,
)


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_clock_fragment(text: str) -> Optional[Tuple[int, int]]:
    """Extract hour/minute from free text like '10 am', '10:30pm', '14:00'."""
    best: Optional[Tuple[int, int, int]] = None
    for m in _CLOCK_RE.finditer(text or ""):
        try:
            hour = int(m.group("h"))
            minute = int(m.group("m") or 0)
        except (TypeError, ValueError):
            continue
        ampm = (m.group("ampm") or "").lower().replace(".", "")
        if ampm.startswith("p") and hour < 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
        elif not ampm:
            if hour > 23:
                continue
            if hour <= 12 and ":" not in m.group(0) and hour != 0:
                pass
        if hour > 23 or minute > 59:
            continue
        span = m.end() - m.start()
        if best is None or span >= best[0]:
            best = (span, hour, minute)
    if best is None:
        return None
    return best[1], best[2]


def _parse_soft_clock_fragment(text: str) -> Optional[Tuple[int, int]]:
    """Map vague parts of day to a clinic clock (for ranking / nearest search)."""
    low = (text or "").lower()
    if re.search(r"\b(noon|mid[\s-]?day)\b", low):
        return 12, 0
    if re.search(r"\bmorning\b", low):
        return 10, 0
    if re.search(r"\bafternoon\b", low):
        return 15, 0  # after default lunch 14:00–15:00
    if re.search(r"\bevening\b", low):
        return 16, 0
    return None


def _resolve_relative_date(text: str, now: datetime):
    low = (text or "").lower()
    today = now.date()
    if "day after tomorrow" in low:
        return today + timedelta(days=2)
    if "tomorrow" in low:
        return today + timedelta(days=1)
    if "today" in low:
        return today
    for name, target in _WEEKDAYS.items():
        if name not in low:
            continue
        days_ahead = (target - today.weekday()) % 7
        if "next" in low and days_ahead == 0:
            days_ahead = 7
        elif days_ahead == 0 and "this" not in low:
            days_ahead = 0
        return today + timedelta(days=days_ahead)
    return None


def resolve_appointment_day(
    raw: str,
    *,
    now: Optional[datetime] = None,
):
    """Calendar day from free text even when no clock is given ('today', '2026-07-18')."""
    s = (raw or "").strip()
    if not s:
        return None
    now = now or datetime.now()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})\b", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            return None
    return _resolve_relative_date(s, now)


def normalize_appointment_time(
    raw: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Convert free-text times to 'YYYY-MM-DD HH:MM' for storage and admin filters.

    Day-only phrases like 'today' return None — use resolve_appointment_day /
    find_nearest_available_times for availability. Soft clocks (morning/afternoon)
    are accepted when a day is present.
    """
    s = (raw or "").strip()
    if not s:
        return None
    now = now or datetime.now()

    for fmt in _APPT_TIME_FMTS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue

    m_iso = re.match(
        r"^(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})(?::\d{2})?$",
        s,
    )
    if m_iso:
        try:
            dt = datetime(
                int(m_iso.group(1)[0:4]),
                int(m_iso.group(1)[5:7]),
                int(m_iso.group(1)[8:10]),
                int(m_iso.group(2)),
                int(m_iso.group(3)),
            )
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    day = resolve_appointment_day(s, now=now)
    clock = _parse_clock_fragment(s) or _parse_soft_clock_fragment(s)
    if day is None or clock is None:
        return None
    hour, minute = clock
    try:
        dt = datetime.combine(day, dt_time(hour=hour, minute=minute))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M")


def parse_appointment_datetime(raw: str, *, now: Optional[datetime] = None) -> Optional[datetime]:
    normalized = normalize_appointment_time(raw, now=now)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def _next_clinic_slot_on_or_after(
    day,
    after: datetime,
    *,
    slot_minutes: int = _DEFAULT_SLOT_MINUTES,
) -> Optional[datetime]:
    """First clinic slot on ``day`` at or after ``after`` (exclusive of past)."""
    for slot in _iter_clinic_slots_on_day(day, slot_minutes=slot_minutes):
        if slot > after:
            return slot
    return None


def parse_availability_anchor(
    raw: str,
    *,
    now: Optional[datetime] = None,
) -> Tuple[Optional[datetime], bool]:
    """Anchor datetime for nearest-slot search.

    Returns (anchor, day_only). day_only=True when the user named a day but no clock
    (e.g. 'today') — the exact anchor slot may still be bookable.
    """
    now = now or datetime.now()
    exact = parse_appointment_datetime(raw, now=now)
    if exact is not None:
        return exact, False
    day = resolve_appointment_day(raw, now=now)
    if day is None:
        return None, False
    after = now if day == now.date() else datetime.combine(day, _CLINIC_START) - timedelta(minutes=1)
    anchor = _next_clinic_slot_on_or_after(day, after)
    if anchor is None:
        # Day exhausted — still return clinic start so search can look ahead.
        return datetime.combine(day, _CLINIC_START), True
    return anchor, True


def _hm_to_minutes(hm: str) -> Optional[int]:
    try:
        parts = (hm or "").strip().split(":")
        if len(parts) < 2:
            return None
        h, m = int(parts[0]), int(parts[1])
        if h < 0 or h > 23 or m < 0 or m > 59:
            return None
        return h * 60 + m
    except (TypeError, ValueError):
        return None


def is_within_clinic_hours(time_str: str) -> bool:
    """True when slot start is in [09:00, 17:00)."""
    dt = parse_appointment_datetime(time_str)
    if dt is None:
        return False
    minutes = dt.hour * 60 + dt.minute
    start = _CLINIC_START.hour * 60 + _CLINIC_START.minute
    end = _CLINIC_END.hour * 60 + _CLINIC_END.minute
    return start <= minutes < end


def _iter_clinic_slots_on_day(day, slot_minutes: int = _DEFAULT_SLOT_MINUTES) -> List[datetime]:
    """All clinic slots on a calendar day (09:00 inclusive … 17:00 exclusive)."""
    slots: List[datetime] = []
    cur = datetime.combine(day, _CLINIC_START)
    end = datetime.combine(day, _CLINIC_END)
    step = timedelta(minutes=max(1, int(slot_minutes)))
    while cur < end:
        slots.append(cur)
        cur += step
    return slots


def classify_appointment_timing(
    time_str: str,
    *,
    now: Optional[datetime] = None,
    soon_hours: float = APPOINTMENT_SOON_HOURS,
) -> Dict[str, Any]:
    """Classify how soon an appointment is relative to now."""
    now = now or datetime.now()
    dt = parse_appointment_datetime(str(time_str or ""), now=now)
    if dt is None:
        return {
            "timing_bucket": "unknown",
            "hours_until": None,
            "appointment_time": time_str,
        }
    delta_sec = (dt - now).total_seconds()
    hours = delta_sec / 3600.0
    if hours < 0:
        bucket = "past"
    elif hours <= float(soon_hours):
        bucket = "soon"
    else:
        bucket = "later"
    return {
        "timing_bucket": bucket,
        "hours_until": round(hours, 2),
        "appointment_time": dt.strftime("%Y-%m-%d %H:%M"),
        "soon_hours_threshold": float(soon_hours),
    }
