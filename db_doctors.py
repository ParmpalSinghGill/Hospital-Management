"""Doctors directory, department matching, and unavailable blocks."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from db_core import (
    DEFAULT_LUNCH_END,
    DEFAULT_LUNCH_START,
    _CLINIC_END,
    _CLINIC_START,
    _DEFAULT_SLOT_MINUTES,
    _db,
    _next_id,
    _row_to_dict,
)
from db_time import (
    _hm_to_minutes,
    _iter_clinic_slots_on_day,
    normalize_appointment_time,
    parse_appointment_datetime,
)

def _normalize_doctor_name(name: str) -> str:
    n = name.strip().lower()
    if n.startswith("dr. "):
        n = n[4:]
    elif n.startswith("dr "):
        n = n[3:]
    return n


_DEPARTMENT_ALIASES = {
    "dental": "dentistry",
    "dentist": "dentistry",
    "dentists": "dentistry",
    "tooth": "dentistry",
    "teeth": "dentistry",
    "oral": "dentistry",
    "heart": "cardiology",
    "cardiac": "cardiology",
    "skin": "dermatology",
    "bone": "orthopedics",
    "ortho": "orthopedics",
    "eye": "ophthalmology",
    "eyes": "ophthalmology",
    "ear": "ent",
    "nose": "ent",
    "throat": "ent",
    "kidney": "nephrology",
    "lung": "pulmonology",
    "lungs": "pulmonology",
    "mental": "psychiatry",
    "psych": "psychiatry",
    "child": "pediatrics",
    "kids": "pediatrics",
    "children": "pediatrics",
    "pregnant": "gynecology",
    "women": "gynecology",
    "general": "general medicine",
    "gp": "family medicine",
    "family": "family medicine",
    "stomach": "gastroenterology",
    "belly": "gastroenterology",
    "abdomen": "gastroenterology",
    "abdominal": "gastroenterology",
    "digestive": "gastroenterology",
    "gastric": "gastroenterology",
    "headache": "general medicine",
    "fever": "general medicine",
    "cold": "general medicine",
    "flu": "general medicine",
}


def _department_query_terms(department: str) -> List[str]:
    """Expand a spoken department filter into match terms."""
    raw = (department or "").strip().lower()
    if not raw:
        return []
    terms = {raw}
    if raw in _DEPARTMENT_ALIASES:
        terms.add(_DEPARTMENT_ALIASES[raw])
    # Also map multi-word queries word-by-word for aliases.
    for word in raw.replace("&", " ").replace("/", " ").split():
        w = word.strip()
        if w in _DEPARTMENT_ALIASES:
            terms.add(_DEPARTMENT_ALIASES[w])
    return list(terms)


def department_matches(filter_dep: str, doctor_dep: str) -> bool:
    """True if filter matches doctor department (aliases + safe prefix matching)."""
    doc = (doctor_dep or "").strip().lower()
    if not (filter_dep or "").strip():
        return True
    if not doc:
        return False
    doc_tokens = [
        t for t in doc.replace("&", " ").replace("/", " ").replace("-", " ").split() if t
    ]
    for term in _department_query_terms(filter_dep):
        if not term:
            continue
        if term == doc or term in doc_tokens:
            return True
        # Prefix only (dental → dentistry). Never bare substring (ent ⊂ dental).
        if len(term) >= 5 and doc.startswith(term):
            return True
        if len(doc) >= 5 and term.startswith(doc):
            return True
        for tok in doc_tokens:
            if len(term) >= 5 and tok.startswith(term):
                return True
            if len(tok) >= 5 and term.startswith(tok):
                return True
    return False


def ensure_default_lunch_breaks() -> int:
    """Ensure every doctor has a recurring 14:00–15:00 lunch block."""
    created = 0
    doctors = _load_doctors()
    with _db() as conn:
        for d in doctors:
            did = str(d.get("doctor_id") or "").strip()
            if not did:
                continue
            row = conn.execute(
                "SELECT block_id FROM doctor_unavailable "
                "WHERE doctor_id=? AND day='' AND start_hm=? AND end_hm=? AND reason='lunch'",
                (did, DEFAULT_LUNCH_START, DEFAULT_LUNCH_END),
            ).fetchone()
            if row:
                continue
            block_id = f"BLK-LUNCH-{did}"
            conn.execute(
                "INSERT OR IGNORE INTO doctor_unavailable("
                "block_id, doctor_id, day, start_hm, end_hm, reason"
                ") VALUES (?,?,?,?,?,?)",
                (block_id, did, "", DEFAULT_LUNCH_START, DEFAULT_LUNCH_END, "lunch"),
            )
            created += 1
    return created


def _load_unavailable_blocks(doctor_id: str, day: str = "") -> List[Dict[str, Any]]:
    did = (doctor_id or "").strip().upper()
    day_key = (day or "").strip()[:10]
    with _db() as conn:
        rows = conn.execute(
            "SELECT block_id, doctor_id, day, start_hm, end_hm, reason "
            "FROM doctor_unavailable WHERE upper(doctor_id)=? "
            "AND (day='' OR day=?)",
            (did, day_key),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def is_doctor_unavailable(
    doctor_id: str,
    time_str: str,
) -> Optional[Dict[str, Any]]:
    """Return the blocking unavailable row if doctor is off at that datetime."""
    dt = parse_appointment_datetime(time_str)
    if dt is None:
        return {"reason": "invalid_time", "start_hm": "", "end_hm": ""}
    day = dt.strftime("%Y-%m-%d")
    slot_m = dt.hour * 60 + dt.minute
    for block in _load_unavailable_blocks(doctor_id, day):
        start_m = _hm_to_minutes(str(block.get("start_hm") or ""))
        end_m = _hm_to_minutes(str(block.get("end_hm") or ""))
        if start_m is None or end_m is None:
            continue
        if start_m <= slot_m < end_m:
            return block
    return None


def _next_unavailable_block_id() -> str:
    with _db() as conn:
        rows = conn.execute("SELECT block_id FROM doctor_unavailable").fetchall()
    return _next_id([r["block_id"] for r in rows], "BLK")


def add_doctor_unavailable(
    doctor_id: str,
    start_hm: str,
    end_hm: str,
    *,
    day: str = "",
    reason: str = "unavailable",
) -> Dict[str, Any]:
    from db_appointments import _load_all_appointments
    did = (doctor_id or "").strip()
    if not did or not _get_doctor_by_id(did):
        raise ValueError(f"Doctor not found: {doctor_id}")
    s = _hm_to_minutes(start_hm)
    e = _hm_to_minutes(end_hm)
    if s is None or e is None or e <= s:
        raise ValueError("start_hm/end_hm must be HH:MM with end after start")
    day_key = (day or "").strip()[:10]
    start_padded = f"{s // 60:02d}:{s % 60:02d}"
    end_padded = f"{e // 60:02d}:{e % 60:02d}"
    # Refuse if active appointments overlap this window on a specific day
    if day_key:
        start_dt = datetime.strptime(f"{day_key} {start_padded}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{day_key} {end_padded}", "%Y-%m-%d %H:%M")
        now = datetime.now()
        if end_dt <= now:
            raise ValueError("Cannot mark unavailable in the past.")
        overlapping = []
        for a in _load_all_appointments():
            if str(a.get("doctor_id") or "").strip().upper() != did.upper():
                continue
            if a.get("status") == "CANCELLED":
                continue
            when = parse_appointment_datetime(str(a.get("time") or ""))
            if when is None:
                continue
            if start_dt <= when < end_dt:
                overlapping.append(a.get("appointment_id"))
        if overlapping:
            raise ValueError(
                f"Cannot mark unavailable: appointments exist in that window "
                f"({', '.join(str(x) for x in overlapping[:5])}). Cancel or move them first."
            )
    block = {
        "block_id": _next_unavailable_block_id(),
        "doctor_id": did,
        "day": day_key,
        "start_hm": start_padded,
        "end_hm": end_padded,
        "reason": (reason or "unavailable").strip() or "unavailable",
    }
    with _db() as conn:
        conn.execute(
            "INSERT INTO doctor_unavailable(block_id, doctor_id, day, start_hm, end_hm, reason) "
            "VALUES (?,?,?,?,?,?)",
            (
                block["block_id"],
                block["doctor_id"],
                block["day"],
                block["start_hm"],
                block["end_hm"],
                block["reason"],
            ),
        )
    return block


def remove_doctor_unavailable(block_id: str) -> bool:
    bid = (block_id or "").strip()
    if not bid:
        return False
    with _db() as conn:
        cur = conn.execute("DELETE FROM doctor_unavailable WHERE block_id=?", (bid,))
        return cur.rowcount > 0


def get_doctor_day_grid(
    doctor_id: str,
    day: str,
    *,
    slot_minutes: int = _DEFAULT_SLOT_MINUTES,
) -> Dict[str, Any]:
    """8×6 clinic grid for one doctor/day: booked, unavailable, or free."""
    from db_appointments import _enrich_appointment, _load_all_appointments
    doc = _get_doctor_by_id(doctor_id)
    if not doc:
        raise ValueError(f"Doctor not found: {doctor_id}")
    day_key = normalize_appointment_time(f"{day} 09:00")
    if not day_key:
        # accept YYYY-MM-DD alone
        try:
            datetime.strptime((day or "").strip()[:10], "%Y-%m-%d")
            day_key = (day or "").strip()[:10]
        except ValueError as exc:
            raise ValueError(f"Invalid day: {day}") from exc
    else:
        day_key = day_key[:10]

    day_date = datetime.strptime(day_key, "%Y-%m-%d").date()
    slots_dt = _iter_clinic_slots_on_day(day_date, slot_minutes=slot_minutes)
    # Bookings that day
    booked_by_time: Dict[str, Dict[str, Any]] = {}
    for a in _load_all_appointments():
        if str(a.get("doctor_id") or "").strip().upper() != str(doctor_id).strip().upper():
            continue
        if a.get("status") == "CANCELLED":
            continue
        norm = normalize_appointment_time(str(a.get("time") or ""))
        if not norm or not norm.startswith(day_key):
            continue
        booked_by_time[norm] = _enrich_appointment(a)

    blocks = _load_unavailable_blocks(doctor_id, day_key)
    now = datetime.now()
    cells: List[Dict[str, Any]] = []
    for slot in slots_dt:
        key = slot.strftime("%Y-%m-%d %H:%M")
        hm = slot.strftime("%H:%M")
        appt = booked_by_time.get(key)
        block = is_doctor_unavailable(doctor_id, key)
        status = "free"
        if appt:
            status = "booked"
        elif block:
            status = "unavailable"
        past = slot < now
        cells.append({
            "time": key,
            "hm": hm,
            "hour": slot.hour,
            "minute": slot.minute,
            "status": status,
            "past": past,
            "editable": not past,
            "appointment": appt,
            "unavailable": block,
        })

    # Shape as 8 rows (hours 9..16) × 6 cols (0,10,20,30,40,50)
    hours = list(range(_CLINIC_START.hour, _CLINIC_END.hour))
    minutes = list(range(0, 60, max(1, int(slot_minutes))))
    grid: List[List[Optional[Dict[str, Any]]]] = []
    by_hm = {c["hm"]: c for c in cells}
    for h in hours:
        row = []
        for m in minutes:
            hm = f"{h:02d}:{m:02d}"
            row.append(by_hm.get(hm))
        grid.append(row)

    free = sum(1 for c in cells if c["status"] == "free" and not c.get("past"))
    booked = sum(1 for c in cells if c["status"] == "booked")
    unavailable = sum(1 for c in cells if c["status"] == "unavailable")
    past_count = sum(1 for c in cells if c.get("past"))
    return {
        "doctor": doc,
        "day": day_key,
        "clinic_start": _CLINIC_START.strftime("%H:%M"),
        "clinic_end": _CLINIC_END.strftime("%H:%M"),
        "slot_minutes": slot_minutes,
        "as_of": now.strftime("%Y-%m-%d %H:%M"),
        "hours": hours,
        "minutes": minutes,
        "cells": cells,
        "grid": grid,
        "blocks": blocks,
        "stats": {
            "total": len(cells),
            "free": free,
            "booked": booked,
            "unavailable": unavailable,
            "past": past_count,
            "fill_pct": round(100.0 * booked / max(1, free + booked), 1),
        },
    }


def _load_doctors() -> List[Dict[str, str]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT doctor_id, name, department FROM doctors ORDER BY doctor_id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _save_doctors(doctors: List[Dict[str, Any]]) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM doctors")
        for i, d in enumerate(doctors):
            did = str(d.get("doctor_id") or f"DOC-{(i + 1):04d}")
            conn.execute(
                "INSERT INTO doctors(doctor_id, name, department) VALUES (?,?,?)",
                (did, d.get("name", ""), d.get("department", "")),
            )


def _get_doctor_by_name(name: str) -> Optional[Dict[str, str]]:
    target = _normalize_doctor_name(name)
    for d in _load_doctors():
        if _normalize_doctor_name(d.get("name", "")) == target:
            return d
    return None


def _get_doctor_by_id(doctor_id: str) -> Optional[Dict[str, str]]:
    target = (doctor_id or "").strip().upper()
    with _db() as conn:
        row = conn.execute(
            "SELECT doctor_id, name, department FROM doctors WHERE upper(doctor_id)=?",
            (target,),
        ).fetchone()
    return _row_to_dict(row) if row else None
