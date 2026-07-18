"""Hospital data layer — SQLite store under dataset/hospital.db.

Public helpers keep the same names Tools.py / admin_routes.py / MakeDataBase.py
already use (_load_*, _save_*, lookup helpers, time normalization).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, time as dt_time
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_PATH = os.path.join(DATASET_DIR, "hospital.db")

# Legacy JSON paths (used only for one-time migration)
DOCTORS_DB = os.path.join(DATASET_DIR, "doctors.json")
PATIENTS_DB = os.path.join(DATASET_DIR, "patients.json")
APPOINTMENTS_DB = os.path.join(DATASET_DIR, "appointments.json")
PRESCRIPTIONS_DB = os.path.join(DATASET_DIR, "prescriptions.json")

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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    phone       TEXT NOT NULL DEFAULT '',
    phone_digits TEXT NOT NULL DEFAULT '',
    address     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    patient_id     TEXT NOT NULL DEFAULT '',
    doctor_id      TEXT NOT NULL DEFAULT '',
    doctor         TEXT NOT NULL DEFAULT '',
    department     TEXT NOT NULL DEFAULT '',
    time           TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'BOOKED'
);

CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id TEXT PRIMARY KEY,
    patient_id      TEXT NOT NULL DEFAULT '',
    doctor_id       TEXT NOT NULL DEFAULT '',
    medicine_name   TEXT NOT NULL DEFAULT '',
    timing          TEXT NOT NULL DEFAULT ''
);

-- Doctor off-blocks. day='' means every clinic day (recurring).
-- Interval is [start_hm, end_hm) in HH:MM. Default lunch is 14:00-15:00.
CREATE TABLE IF NOT EXISTS doctor_unavailable (
    block_id   TEXT PRIMARY KEY,
    doctor_id  TEXT NOT NULL,
    day        TEXT NOT NULL DEFAULT '',
    start_hm   TEXT NOT NULL,
    end_hm     TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT 'unavailable'
);

CREATE INDEX IF NOT EXISTS ix_patients_phone_digits ON patients(phone_digits);
CREATE INDEX IF NOT EXISTS ix_appointments_patient ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS ix_appointments_doctor_time ON appointments(doctor_id, time);
CREATE INDEX IF NOT EXISTS ix_appointments_status ON appointments(status);
CREATE INDEX IF NOT EXISTS ix_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX IF NOT EXISTS ix_unavailable_doctor ON doctor_unavailable(doctor_id, day);

-- One active booking per doctor+time (cancelled rows ignored).
CREATE UNIQUE INDEX IF NOT EXISTS ux_active_doctor_time
ON appointments(doctor_id, time)
WHERE status != 'CANCELLED' AND doctor_id != '' AND time != '';
"""

# Clinic hours: doctors are only bookable 09:00–17:00 (not 24h).
DEFAULT_LUNCH_START = "14:00"
DEFAULT_LUNCH_END = "15:00"
_CLINIC_START = dt_time(9, 0)
_CLINIC_END = dt_time(17, 0)
_DEFAULT_SLOT_MINUTES = 10

_initialized = False
# Serialize appointment mutations in-process so load-modify-save cannot
# wipe rows written by a concurrent book/cancel/reschedule.
_appointment_write_lock = threading.Lock()


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def _normalize_doctor_name(name: str) -> str:
    n = name.strip().lower()
    if n.startswith("dr. "):
        n = n[4:]
    elif n.startswith("dr "):
        n = n[3:]
    return n


# Common spoken aliases → directory department names (substring match still applies).
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


def _normalize_phone(phone: str) -> str:
    return "".join(ch for ch in (phone or "") if ch.isdigit())


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


def normalize_appointment_time(
    raw: str,
    *,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Convert free-text times to 'YYYY-MM-DD HH:MM' for storage and admin filters."""
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

    day = _resolve_relative_date(s, now)
    clock = _parse_clock_fragment(s)
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


def _extract_id_number(record_id: str, prefix: str) -> int:
    try:
        raw = record_id
        if raw.upper().startswith(prefix.upper()):
            raw = raw[len(prefix) :]
            if raw.startswith("-"):
                raw = raw[1:]
        return int(raw)
    except Exception:
        return 0


def _next_id(existing_ids: List[str], prefix: str, width: int = 4) -> str:
    max_num = 0
    for rid in existing_ids:
        max_num = max(max_num, _extract_id_number(str(rid), prefix))
    return f"{prefix}-{(max_num + 1):0{width}d}"


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create schema and migrate legacy JSON once if the DB is empty."""
    global _initialized
    if _initialized and os.path.exists(DB_PATH):
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        empty = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 0
        empty = empty and conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0
        empty = empty and conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
        if empty:
            _migrate_json_into(conn)
            conn.commit()
    finally:
        conn.close()
    _initialized = True
    ensure_default_lunch_breaks()


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


def check_slot_bookable(
    doctor_id: str,
    time_str: str,
    *,
    exclude_appointment_id: str = "",
    doctor_name: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a refusal payload if the slot cannot be booked; None if OK."""
    norm = normalize_appointment_time(time_str)
    if not norm:
        return {
            "ok": False,
            "reason": "invalid_time",
            "message": f"Could not understand the appointment time '{time_str}'.",
        }
    if not is_within_clinic_hours(norm):
        return {
            "ok": False,
            "reason": "outside_clinic_hours",
            "message": (
                f"Doctors are only available from 9:00 AM to 5:00 PM. "
                f"'{norm}' is outside clinic hours."
            ),
            "requested_time": norm,
        }
    block = is_doctor_unavailable(doctor_id, norm)
    if block:
        reason = block.get("reason") or "unavailable"
        return {
            "ok": False,
            "reason": "doctor_unavailable",
            "message": (
                f"Doctor is not available at {norm} "
                f"({reason}: {block.get('start_hm')}–{block.get('end_hm')}). "
                "Choose another time."
            ),
            "requested_time": norm,
            "unavailable": block,
        }
    conflict = _find_conflict(
        None,
        doctor_name,
        norm,
        doctor_id=doctor_id,
        exclude_appointment_id=exclude_appointment_id,
    )
    if conflict:
        return {
            "ok": False,
            "reason": "time_conflict",
            "message": f"Doctor already has an appointment at '{norm}'.",
            "requested_time": norm,
            "conflict": conflict,
        }
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
        cells.append({
            "time": key,
            "hm": hm,
            "hour": slot.hour,
            "minute": slot.minute,
            "status": status,
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

    free = sum(1 for c in cells if c["status"] == "free")
    booked = sum(1 for c in cells if c["status"] == "booked")
    unavailable = sum(1 for c in cells if c["status"] == "unavailable")
    return {
        "doctor": doc,
        "day": day_key,
        "clinic_start": _CLINIC_START.strftime("%H:%M"),
        "clinic_end": _CLINIC_END.strftime("%H:%M"),
        "slot_minutes": slot_minutes,
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
            "fill_pct": round(100.0 * booked / max(1, free + booked), 1),
        },
    }


def _migrate_json_into(conn: sqlite3.Connection) -> None:
    """Import dataset/*.json into SQLite when present (one-time)."""
    doctors = _read_json(DOCTORS_DB, {"doctors": []}).get("doctors", [])
    for i, d in enumerate(doctors):
        did = str(d.get("doctor_id") or f"DOC-{(i + 1):04d}")
        conn.execute(
            "INSERT OR REPLACE INTO doctors(doctor_id, name, department) VALUES (?,?,?)",
            (did, d.get("name", ""), d.get("department", "")),
        )

    for p in _read_json(PATIENTS_DB, {"patients": []}).get("patients", []):
        phone = str(p.get("phone") or "")
        conn.execute(
            "INSERT OR REPLACE INTO patients(patient_id, name, phone, phone_digits, address) "
            "VALUES (?,?,?,?,?)",
            (
                p.get("patient_id", ""),
                p.get("name", ""),
                phone,
                _normalize_phone(phone),
                p.get("address", ""),
            ),
        )

    for a in _read_json(APPOINTMENTS_DB, {"appointments": []}).get("appointments", []):
        conn.execute(
            "INSERT OR REPLACE INTO appointments("
            "appointment_id, patient_id, doctor_id, doctor, department, time, status"
            ") VALUES (?,?,?,?,?,?,?)",
            (
                a.get("appointment_id", ""),
                a.get("patient_id", ""),
                a.get("doctor_id", ""),
                a.get("doctor", ""),
                a.get("department", ""),
                a.get("time", ""),
                a.get("status", "BOOKED"),
            ),
        )

    for r in _read_json(PRESCRIPTIONS_DB, {"prescriptions": []}).get("prescriptions", []):
        conn.execute(
            "INSERT OR REPLACE INTO prescriptions("
            "prescription_id, patient_id, doctor_id, medicine_name, timing"
            ") VALUES (?,?,?,?,?)",
            (
                r.get("prescription_id", ""),
                r.get("patient_id", ""),
                r.get("doctor_id", ""),
                r.get("medicine_name", ""),
                r.get("timing", ""),
            ),
        )


# -------- Doctors --------
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


# -------- Patients --------
def _patient_public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patient_id": row.get("patient_id", ""),
        "name": row.get("name", ""),
        "phone": row.get("phone", ""),
        "address": row.get("address", ""),
    }


def _load_patients() -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT patient_id, name, phone, address FROM patients ORDER BY patient_id"
        ).fetchall()
    return [_patient_public(_row_to_dict(r)) for r in rows]


def _save_patients(patients: List[Dict[str, Any]]) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM patients")
        for p in patients:
            phone = str(p.get("phone") or "")
            conn.execute(
                "INSERT INTO patients(patient_id, name, phone, phone_digits, address) "
                "VALUES (?,?,?,?,?)",
                (
                    p.get("patient_id", ""),
                    p.get("name", ""),
                    phone,
                    _normalize_phone(phone),
                    p.get("address", ""),
                ),
            )


def _get_next_patient_id(patients: Optional[List[Dict[str, Any]]] = None) -> str:
    if patients is not None:
        return _next_id([p.get("patient_id", "") for p in patients], "PAT")
    with _db() as conn:
        rows = conn.execute("SELECT patient_id FROM patients").fetchall()
    return _next_id([r["patient_id"] for r in rows], "PAT")


def _get_patient_by_id(patient_id: str) -> Optional[Dict[str, Any]]:
    target = (patient_id or "").strip().upper()
    with _db() as conn:
        row = conn.execute(
            "SELECT patient_id, name, phone, address FROM patients WHERE upper(patient_id)=?",
            (target,),
        ).fetchone()
    return _patient_public(_row_to_dict(row)) if row else None


def _find_patient_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    target = _normalize_phone(phone)
    if not target:
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT patient_id, name, phone, address FROM patients WHERE phone_digits=?",
            (target,),
        ).fetchone()
    return _patient_public(_row_to_dict(row)) if row else None


def _find_patient_by_name(name: str) -> Optional[Dict[str, Any]]:
    target = (name or "").strip().lower()
    if not target:
        return None
    with _db() as conn:
        rows = conn.execute(
            "SELECT patient_id, name, phone, address FROM patients "
            "WHERE lower(trim(name))=?",
            (target,),
        ).fetchall()
    if len(rows) == 1:
        return _patient_public(_row_to_dict(rows[0]))
    return None


def _names_match(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _resolve_patient(
    patient_id: Optional[str] = None,
    phone: Optional[str] = None,
    patient_name: Optional[str] = None,
    *,
    require_name_with_phone: bool = False,
) -> Dict[str, Any]:
    """Resolve a patient record with optional name+phone verification."""
    pid = (patient_id or "").strip()
    ph = (phone or "").strip()
    name = (patient_name or "").strip()

    if pid:
        patient = _get_patient_by_id(pid)
        if patient is None:
            return {
                "ok": False,
                "message": f"No patient found for id {pid}.",
                "verified": False,
                "name_mismatch": False,
            }
        if name and not _names_match(name, str(patient.get("name", ""))):
            return {
                "ok": False,
                "patient": patient,
                "message": (
                    f"Patient id {pid} is on file as {patient.get('name')}, "
                    f"which does not match the name given ({name})."
                ),
                "verified": False,
                "name_mismatch": True,
            }
        if ph and _normalize_phone(ph) != _normalize_phone(str(patient.get("phone", ""))):
            return {
                "ok": False,
                "patient": patient,
                "message": "Phone number does not match this patient id.",
                "verified": False,
                "name_mismatch": False,
            }
        return {
            "ok": True,
            "patient": patient,
            "message": "Patient found by id.",
            "verified": True,
            "name_mismatch": False,
            "is_returning": True,
        }

    if require_name_with_phone and (not ph or not name):
        missing = []
        if not name:
            missing.append("name")
        if not ph:
            missing.append("phone")
        return {
            "ok": False,
            "message": (
                "Patient id unknown — please provide both name and phone to verify. "
                f"Missing: {', '.join(missing)}."
            ),
            "verified": False,
            "name_mismatch": False,
            "needs_name": not bool(name),
            "needs_phone": not bool(ph),
        }

    if ph:
        patient = _find_patient_by_phone(ph)
        if patient is None:
            return {
                "ok": True,
                "patient": None,
                "message": "No existing patient with this phone — treat as new patient.",
                "verified": False,
                "name_mismatch": False,
                "is_returning": False,
                "is_new": True,
            }
        if name and not _names_match(name, str(patient.get("name", ""))):
            return {
                "ok": False,
                "patient": patient,
                "message": (
                    f"Phone matches patient {patient.get('patient_id')} "
                    f"({patient.get('name')}), but the name given ({name}) does not match. "
                    "Please confirm the correct name."
                ),
                "verified": False,
                "name_mismatch": True,
                "is_returning": True,
            }
        verified = bool(name) or not require_name_with_phone
        return {
            "ok": True,
            "patient": patient,
            "message": "Existing patient found by phone.",
            "verified": verified,
            "name_mismatch": False,
            "is_returning": True,
            "is_new": False,
            "needs_name": not bool(name),
        }

    if name and not require_name_with_phone:
        patient = _find_patient_by_name(name)
        if patient is None:
            return {
                "ok": False,
                "message": "No patient found with that exact name. Ask for phone number.",
                "verified": False,
                "name_mismatch": False,
                "needs_phone": True,
            }
        return {
            "ok": True,
            "patient": patient,
            "message": "Patient found by name; confirm with phone if possible.",
            "verified": False,
            "name_mismatch": False,
            "is_returning": True,
            "needs_phone": True,
        }

    return {
        "ok": False,
        "message": "Provide patient_id, or phone (and name when verifying without id).",
        "verified": False,
        "name_mismatch": False,
        "needs_phone": True,
    }


def _patient_past_doctors(patient_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Unique doctors this patient has seen, most recent appointment first."""
    pid = (patient_id or "").strip()
    if not pid:
        return []

    visits: List[Dict[str, Any]] = []
    for a in _load_all_appointments():
        if a.get("patient_id") != pid:
            continue
        if a.get("status") == "CANCELLED":
            continue
        doctor = None
        if a.get("doctor_id"):
            doctor = _get_doctor_by_id(str(a.get("doctor_id")))
        if doctor is None and a.get("doctor"):
            doctor = _get_doctor_by_name(str(a.get("doctor")))
        if not doctor:
            continue
        visits.append({
            "doctor_id": doctor.get("doctor_id"),
            "doctor": doctor.get("name"),
            "department": doctor.get("department", a.get("department", "")),
            "last_visit": a.get("time", ""),
            "appointment_id": a.get("appointment_id"),
        })

    seen_ids = {v.get("doctor_id") for v in visits if v.get("doctor_id")}
    for r in _load_prescriptions():
        if r.get("patient_id") != pid:
            continue
        did = r.get("doctor_id")
        if not did or did in seen_ids:
            continue
        doctor = _get_doctor_by_id(str(did))
        if not doctor:
            continue
        visits.append({
            "doctor_id": doctor.get("doctor_id"),
            "doctor": doctor.get("name"),
            "department": doctor.get("department", ""),
            "last_visit": "",
            "from_prescription": True,
        })
        seen_ids.add(did)

    visits.sort(key=lambda v: str(v.get("last_visit") or ""), reverse=True)

    unique: List[Dict[str, Any]] = []
    seen: set = set()
    for v in visits:
        key = v.get("doctor_id") or v.get("doctor")
        if key in seen:
            continue
        seen.add(key)
        unique.append(v)
        if len(unique) >= limit:
            break
    return unique


def _get_or_create_patient(
    name: str,
    phone: str,
    address: str = "",
) -> Dict[str, Any]:
    """Find patient by phone (preferred); otherwise create. Phone is the unique key."""
    existing = _find_patient_by_phone(phone) if phone else None
    if existing is not None:
        changed = False
        if name and _names_match(name, str(existing.get("name", ""))):
            pass
        elif name and not existing.get("name"):
            existing["name"] = name.strip()
            changed = True
        if address and not existing.get("address"):
            existing["address"] = address.strip()
            changed = True
        if changed:
            with _db() as conn:
                conn.execute(
                    "UPDATE patients SET name=?, address=? WHERE patient_id=?",
                    (
                        existing.get("name", ""),
                        existing.get("address", ""),
                        existing.get("patient_id"),
                    ),
                )
        return existing

    new_patient = {
        "patient_id": _get_next_patient_id(),
        "name": name.strip(),
        "phone": (phone or "").strip(),
        "address": (address or "").strip(),
    }
    with _db() as conn:
        conn.execute(
            "INSERT INTO patients(patient_id, name, phone, phone_digits, address) "
            "VALUES (?,?,?,?,?)",
            (
                new_patient["patient_id"],
                new_patient["name"],
                new_patient["phone"],
                _normalize_phone(new_patient["phone"]),
                new_patient["address"],
            ),
        )
    return new_patient


# -------- Appointments --------
def _load_all_appointments() -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT appointment_id, patient_id, doctor_id, doctor, department, time, status "
            "FROM appointments ORDER BY appointment_id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _save_all_appointments(appts: List[Dict[str, Any]]) -> None:
    """Bulk replace used by MakeDataBase seeding only — not by live tools."""
    with _appointment_write_lock:
        with _db() as conn:
            conn.execute("DELETE FROM appointments")
            for a in appts:
                conn.execute(
                    "INSERT INTO appointments("
                    "appointment_id, patient_id, doctor_id, doctor, department, time, status"
                    ") VALUES (?,?,?,?,?,?,?)",
                    (
                        a.get("appointment_id", ""),
                        a.get("patient_id", ""),
                        a.get("doctor_id", ""),
                        a.get("doctor", ""),
                        a.get("department", ""),
                        a.get("time", ""),
                        a.get("status", "BOOKED"),
                    ),
                )


def _insert_appointment(appt: Dict[str, Any]) -> None:
    """Insert a single appointment row (live booking path)."""
    with _appointment_write_lock:
        with _db() as conn:
            conn.execute(
                "INSERT INTO appointments("
                "appointment_id, patient_id, doctor_id, doctor, department, time, status"
                ") VALUES (?,?,?,?,?,?,?)",
                (
                    appt.get("appointment_id", ""),
                    appt.get("patient_id", ""),
                    appt.get("doctor_id", ""),
                    appt.get("doctor", ""),
                    appt.get("department", ""),
                    appt.get("time", ""),
                    appt.get("status", "BOOKED"),
                ),
            )


def _update_appointment(appt: Dict[str, Any]) -> None:
    """Update one appointment by id (cancel / reschedule). Never rewrites the table."""
    aid = str(appt.get("appointment_id") or "").strip()
    if not aid:
        raise ValueError("appointment_id required for update")
    with _appointment_write_lock:
        with _db() as conn:
            cur = conn.execute(
                "UPDATE appointments SET patient_id=?, doctor_id=?, doctor=?, "
                "department=?, time=?, status=? WHERE appointment_id=?",
                (
                    appt.get("patient_id", ""),
                    appt.get("doctor_id", ""),
                    appt.get("doctor", ""),
                    appt.get("department", ""),
                    appt.get("time", ""),
                    appt.get("status", "BOOKED"),
                    aid,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Appointment not found for update: {aid}")


def _get_next_appointment_id(appts: Optional[List[Dict[str, Any]]] = None) -> str:
    if appts is not None:
        return _next_id([a.get("appointment_id", "") for a in appts], "APT")
    with _db() as conn:
        rows = conn.execute("SELECT appointment_id FROM appointments").fetchall()
    return _next_id([r["appointment_id"] for r in rows], "APT")


def _find_conflict(
    appts: Optional[List[Dict[str, Any]]],
    doctor_name: str,
    time_str: str,
    doctor_id: str = "",
    exclude_appointment_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Return another active appointment for the same doctor at the same normalized time."""
    doc_norm = _normalize_doctor_name(doctor_name) if doctor_name else ""
    doc_id = (doctor_id or "").strip().upper()
    if not doc_id and doctor_name:
        row = _get_doctor_by_name(doctor_name)
        if row:
            doc_id = str(row.get("doctor_id") or "").strip().upper()
            if not doc_norm:
                doc_norm = _normalize_doctor_name(str(row.get("name") or ""))

    req_norm = normalize_appointment_time(time_str)
    if not req_norm:
        req_norm = (time_str or "").strip().lower()
    if not req_norm:
        return None

    exclude = (exclude_appointment_id or "").strip().upper()
    rows = appts if appts is not None else _load_all_appointments()

    for a in rows:
        if a.get("status") == "CANCELLED":
            continue
        if exclude and str(a.get("appointment_id", "")).upper() == exclude:
            continue

        a_doc_id = str(a.get("doctor_id") or "").strip().upper()
        a_doc = a.get("doctor") or ""
        if not a_doc and a_doc_id:
            doc_row = _get_doctor_by_id(a_doc_id)
            a_doc = (doc_row or {}).get("name", "")
        if not a_doc_id and a_doc:
            doc_row = _get_doctor_by_name(str(a_doc))
            if doc_row:
                a_doc_id = str(doc_row.get("doctor_id") or "").strip().upper()

        same_doctor = False
        if doc_id and a_doc_id and doc_id == a_doc_id:
            same_doctor = True
        elif doc_norm and _normalize_doctor_name(str(a_doc)) == doc_norm:
            same_doctor = True
        if not same_doctor:
            continue

        a_norm = normalize_appointment_time(str(a.get("time", ""))) or str(a.get("time", "")).strip().lower()
        if a_norm == req_norm:
            return a
    return None


def _active_booked_times_for_doctor(
    doctor_id: str,
    *,
    exclude_appointment_id: str = "",
) -> set[str]:
    """Normalized times already taken by this doctor (non-cancelled)."""
    doc_id = (doctor_id or "").strip().upper()
    exclude = (exclude_appointment_id or "").strip().upper()
    taken: set[str] = set()
    if not doc_id:
        return taken
    for a in _load_all_appointments():
        if a.get("status") == "CANCELLED":
            continue
        if exclude and str(a.get("appointment_id", "")).upper() == exclude:
            continue
        if str(a.get("doctor_id") or "").strip().upper() != doc_id:
            continue
        norm = normalize_appointment_time(str(a.get("time", "")))
        if norm:
            taken.add(norm)
    return taken


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


def find_nearest_available_times(
    doctor_id: str,
    preferred_time: str,
    *,
    exclude_appointment_id: str = "",
    limit: int = 3,
    slot_minutes: int = _DEFAULT_SLOT_MINUTES,
    search_days: int = 3,
    now: Optional[datetime] = None,
) -> List[str]:
    """Nearest free slots for this doctor around the requested time (same clinic hours)."""
    preferred = parse_appointment_datetime(preferred_time, now=now)
    if preferred is None:
        return []
    doc_id = (doctor_id or "").strip().upper()
    if not doc_id:
        return []

    taken = _active_booked_times_for_doctor(
        doc_id, exclude_appointment_id=exclude_appointment_id
    )
    now = now or datetime.now()
    candidates: List[Tuple[timedelta, datetime]] = []
    for day_offset in range(0, max(1, int(search_days))):
        day = preferred.date() + timedelta(days=day_offset)
        for slot in _iter_clinic_slots_on_day(day, slot_minutes=slot_minutes):
            if slot <= now:
                continue
            key = slot.strftime("%Y-%m-%d %H:%M")
            if key in taken or key == preferred.strftime("%Y-%m-%d %H:%M"):
                continue
            if is_doctor_unavailable(doc_id, key):
                continue
            candidates.append((abs(slot - preferred), slot))

    candidates.sort(key=lambda item: (item[0], item[1]))
    out: List[str] = []
    seen: set[str] = set()
    for _, slot in candidates:
        key = slot.strftime("%Y-%m-%d %H:%M")
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= max(1, int(limit)):
            break
    return out


def find_available_doctors_at_time(
    time_str: str,
    *,
    department: str = "",
    exclude_doctor_id: str = "",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Doctors free at the requested time, ranked by fewest appointments that day.

    Only doctors who are free at ``time_str`` are returned. Among those, prefer
    the lightest schedule for that calendar day (then by name for stability).
    """
    req = normalize_appointment_time(time_str)
    if not req:
        return []
    day_prefix = req[:10]  # YYYY-MM-DD
    dep = (department or "").strip().lower()
    exclude = (exclude_doctor_id or "").strip().upper()
    day_loads = _appointment_counts_for_day(day_prefix)

    free: List[Dict[str, Any]] = []
    for d in _load_doctors():
        did = str(d.get("doctor_id") or "").strip().upper()
        if not did or did == exclude:
            continue
        if dep and dep not in str(d.get("department") or "").strip().lower():
            continue
        conflict = _find_conflict(
            None,
            str(d.get("name") or ""),
            req,
            doctor_id=did,
        )
        if conflict:
            continue
        if is_doctor_unavailable(did, req):
            continue
        free.append({
            "doctor_id": d.get("doctor_id", ""),
            "name": d.get("name", ""),
            "department": d.get("department", ""),
            "free_at_preferred_time": True,
            "day_appointment_count": int(day_loads.get(did, 0)),
            "day": day_prefix,
        })

    free.sort(
        key=lambda row: (
            int(row.get("day_appointment_count") or 0),
            str(row.get("name") or "").lower(),
        )
    )
    return free[: max(1, int(limit))]


def _appointment_counts_for_day(day_prefix: str) -> Dict[str, int]:
    """Active appointment counts per doctor_id for YYYY-MM-DD."""
    day = (day_prefix or "").strip()[:10]
    counts: Dict[str, int] = {}
    if len(day) < 10:
        return counts
    for a in _load_all_appointments():
        if a.get("status") == "CANCELLED":
            continue
        did = str(a.get("doctor_id") or "").strip().upper()
        if not did:
            continue
        norm = normalize_appointment_time(str(a.get("time", ""))) or str(a.get("time", ""))
        if not str(norm).startswith(day):
            continue
        counts[did] = counts.get(did, 0) + 1
    return counts


def rank_doctors_for_preferred_time(
    doctors: List[Dict[str, Any]],
    preferred_time: str,
) -> List[Dict[str, Any]]:
    """Rank doctors: free at preferred time first, then fewer appointments that day."""
    req = normalize_appointment_time(preferred_time)
    if not req:
        return list(doctors)
    day_prefix = req[:10]
    day_loads = _appointment_counts_for_day(day_prefix)
    ranked: List[Dict[str, Any]] = []
    for d in doctors:
        did = str(d.get("doctor_id") or "").strip().upper()
        conflict = _find_conflict(
            None,
            str(d.get("name") or ""),
            req,
            doctor_id=did,
        )
        free_now = conflict is None and not is_doctor_unavailable(did, req) and is_within_clinic_hours(req)
        row = dict(d)
        row["free_at_preferred_time"] = free_now
        row["day_appointment_count"] = int(day_loads.get(did, 0))
        row["day"] = day_prefix
        ranked.append(row)
    ranked.sort(
        key=lambda row: (
            0 if row.get("free_at_preferred_time") else 1,
            int(row.get("day_appointment_count") or 0),
            str(row.get("name") or "").lower(),
        )
    )
    return ranked


def build_availability_suggestions(
    doctor: Dict[str, Any],
    preferred_time: str,
    *,
    exclude_appointment_id: str = "",
) -> Dict[str, Any]:
    """Suggestions when the preferred doctor+time is taken. Never steals another booking."""
    doc_id = str(doctor.get("doctor_id") or "")
    doc_name = str(doctor.get("name") or doctor.get("doctor") or doc_id)
    department = str(doctor.get("department") or "")
    req = normalize_appointment_time(preferred_time) or preferred_time

    nearest = find_nearest_available_times(
        doc_id,
        req,
        exclude_appointment_id=exclude_appointment_id,
        limit=3,
    )
    alternates = find_available_doctors_at_time(
        req,
        department=department,
        exclude_doctor_id=doc_id,
        limit=5,
    )

    parts = [
        f"{doc_name} is already booked at {req}. "
        "That other patient's appointment stays — we will not replace it."
    ]
    if nearest:
        parts.append(
            f"Nearest open times with {doc_name}: {', '.join(nearest)}. "
            "Offer the closest first; if they accept, book/reschedule that time."
        )
    else:
        parts.append(f"No nearby open times found for {doc_name} in the next few clinic days.")
    if alternates:
        bits = []
        for d in alternates[:3]:
            name = d.get("name") or ""
            load = d.get("day_appointment_count")
            if name and load is not None:
                bits.append(f"{name} ({load} that day)")
            elif name:
                bits.append(name)
        parts.append(
            f"If they insist on {req}, suggest these doctors free at that time "
            f"(least busy that day first): {', '.join(bits)}. "
            f"Confirm a doctor, then book that doctor at {req}."
        )
    else:
        parts.append(
            f"No other doctor in {department or 'that department'} is free at {req}."
        )

    return {
        "requested_time": req,
        "doctor": {"doctor_id": doc_id, "name": doc_name, "department": department},
        "nearest_times": nearest,
        "alternate_doctors": alternates,
        "message": " ".join(parts),
    }


def _enrich_appointment(appt: Dict[str, Any]) -> Dict[str, Any]:
    """Attach patient/doctor display fields for tool responses."""
    out = dict(appt)
    patient = _get_patient_by_id(str(appt.get("patient_id", "")))
    if patient:
        out["patient_name"] = patient.get("name", "")
        out["patient_phone"] = patient.get("phone", "")
    doctor = None
    if appt.get("doctor_id"):
        doctor = _get_doctor_by_id(str(appt.get("doctor_id")))
    if doctor is None and appt.get("doctor"):
        doctor = _get_doctor_by_name(str(appt.get("doctor")))
    if doctor:
        out["doctor"] = doctor.get("name", out.get("doctor", ""))
        out["doctor_id"] = doctor.get("doctor_id", out.get("doctor_id", ""))
        out["department"] = doctor.get("department", out.get("department", ""))
    return out


def _find_active_appointment_for_patient(
    patient_id: str,
    appts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Return the patient's non-cancelled appointment, if any (at most one expected)."""
    pid = (patient_id or "").strip()
    if not pid:
        return None
    if appts is not None:
        for a in appts:
            if a.get("patient_id") == pid and a.get("status") != "CANCELLED":
                return a
        return None
    with _db() as conn:
        row = conn.execute(
            "SELECT appointment_id, patient_id, doctor_id, doctor, department, time, status "
            "FROM appointments WHERE patient_id=? AND status!='CANCELLED' "
            "ORDER BY time DESC LIMIT 1",
            (pid,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def _patient_ids_with_active_appointment(
    appts: Optional[List[Dict[str, Any]]] = None,
) -> set:
    rows = appts if appts is not None else _load_all_appointments()
    return {
        str(a.get("patient_id"))
        for a in rows
        if a.get("patient_id") and a.get("status") != "CANCELLED"
    }


# -------- Prescriptions --------
def _load_prescriptions() -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT prescription_id, patient_id, doctor_id, medicine_name, timing "
            "FROM prescriptions ORDER BY prescription_id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _save_prescriptions(prescriptions: List[Dict[str, Any]]) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM prescriptions")
        for r in prescriptions:
            conn.execute(
                "INSERT INTO prescriptions("
                "prescription_id, patient_id, doctor_id, medicine_name, timing"
                ") VALUES (?,?,?,?,?)",
                (
                    r.get("prescription_id", ""),
                    r.get("patient_id", ""),
                    r.get("doctor_id", ""),
                    r.get("medicine_name", ""),
                    r.get("timing", ""),
                ),
            )


def _get_next_prescription_id(prescriptions: Optional[List[Dict[str, Any]]] = None) -> str:
    if prescriptions is not None:
        return _next_id([p.get("prescription_id", "") for p in prescriptions], "RX")
    with _db() as conn:
        rows = conn.execute("SELECT prescription_id FROM prescriptions").fetchall()
    return _next_id([r["prescription_id"] for r in rows], "RX")


def _find_prescriptions(
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    phone: Optional[str] = None,
) -> Dict[str, Any]:
    """Find prescriptions after verifying the patient."""
    resolved = _resolve_patient(
        patient_id=patient_id,
        phone=phone,
        patient_name=patient_name,
        require_name_with_phone=not bool((patient_id or "").strip()),
    )
    if not resolved.get("ok") or not resolved.get("patient"):
        return {
            "ok": False,
            "message": resolved.get("message") or "Patient not found.",
            "prescriptions": [],
            "needs_name": resolved.get("needs_name", False),
            "needs_phone": resolved.get("needs_phone", False),
            "name_mismatch": resolved.get("name_mismatch", False),
        }

    patient = resolved["patient"]
    pid = patient.get("patient_id")
    rows = [r for r in _load_prescriptions() if r.get("patient_id") == pid]
    enriched = []
    for r in rows:
        item = dict(r)
        item["patient_name"] = patient.get("name", "")
        doctor = _get_doctor_by_id(str(r.get("doctor_id", "")))
        if doctor:
            item["doctor_name"] = doctor.get("name", "")
            item["department"] = doctor.get("department", "")
        enriched.append(item)

    if not enriched:
        return {
            "ok": False,
            "message": "No prescriptions on file for this patient.",
            "patient": patient,
            "prescriptions": [],
        }

    return {
        "ok": True,
        "message": f"Found {len(enriched)} prescription(s).",
        "patient": patient,
        "prescriptions": enriched,
        "count": len(enriched),
    }


def clear_table(table: str) -> int:
    """Delete all rows from a table. Returns previous row count."""
    allowed = {"doctors", "patients", "appointments", "prescriptions"}
    if table not in allowed:
        raise ValueError(f"Unknown table: {table}")
    with _db() as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DELETE FROM {table}")
    return int(count)
