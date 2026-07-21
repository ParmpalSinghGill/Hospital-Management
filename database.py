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

# Legacy JSON paths (optional one-time import if present; live store is hospital.db)
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

-- One active booking per doctor+time (cancelled/completed rows ignored).
CREATE UNIQUE INDEX IF NOT EXISTS ux_active_doctor_time
ON appointments(doctor_id, time)
WHERE status NOT IN ('CANCELLED', 'COMPLETED') AND doctor_id != '' AND time != '';
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

# Active appointment within this many hours → "soon" (inform only, do not rebook).
APPOINTMENT_SOON_HOURS = 12



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


# Indian mobiles are 10 digits; reject shorter lookups so partial STT digits
# (e.g. "91382292") do not create duplicate patients or miss +91 matches.
MIN_PHONE_DIGITS = 10


def _phone_is_complete(phone: str) -> bool:
    return len(_normalize_phone(phone)) >= MIN_PHONE_DIGITS


def _phone_national_digits(phone: str, *, national_len: int = 10) -> str:
    """Return the national subscriber number (last N digits), or all digits if shorter."""
    digits = _normalize_phone(phone)
    if len(digits) <= national_len:
        return digits
    return digits[-national_len:]


def _phone_lookup_candidates(phone: str) -> List[str]:
    """Digit forms that should match the same handset (with/without country code)."""
    target = _normalize_phone(phone)
    if not target:
        return []
    national = _phone_national_digits(target)
    ordered: List[str] = []
    for form in (target, "91" + national if len(national) == 10 else "", national, "0" + national if len(national) == 10 else ""):
        if form and form not in ordered:
            ordered.append(form)
    return ordered


def _phones_match(a: str, b: str, *, min_len: int = 8) -> bool:
    """True when two phones refer to the same number ignoring country/trunk prefix."""
    da, db = _normalize_phone(a), _normalize_phone(b)
    if not da or not db:
        return False
    if da == db:
        return True
    # Prefer last-10 (Indian mobile) when both are long enough
    if len(da) >= 10 and len(db) >= 10:
        return da[-10:] == db[-10:]
    shorter, longer = (da, db) if len(da) <= len(db) else (db, da)
    if len(shorter) >= min_len:
        return longer.endswith(shorter)
    return False


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


_APPOINTMENT_TERMINAL_STATUSES = frozenset({"CANCELLED", "COMPLETED"})


def _appointment_status_is_live(status: str) -> bool:
    """True when the row can block a new booking or counts as an active visit."""
    return (status or "BOOKED") not in _APPOINTMENT_TERMINAL_STATUSES


def _appointment_is_upcoming(
    appt: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> bool:
    """True for future/soon visits that still need patient action."""
    if not _appointment_status_is_live(str(appt.get("status") or "")):
        return False
    timing = classify_appointment_timing(str(appt.get("time") or ""), now=now)
    return timing.get("timing_bucket") != "past"


def _migrate_appointment_unique_index(conn: sqlite3.Connection) -> None:
    """Recreate partial unique index so COMPLETED visits free the slot."""
    conn.execute("DROP INDEX IF EXISTS ux_active_doctor_time")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_active_doctor_time "
        "ON appointments(doctor_id, time) "
        "WHERE status NOT IN ('CANCELLED', 'COMPLETED') "
        "AND doctor_id != '' AND time != ''"
    )


def init_db() -> None:
    """Create schema and migrate legacy JSON once if the DB is empty."""
    global _initialized
    if _initialized:
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.executescript(_SCHEMA)
        _migrate_appointment_unique_index(conn)
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
        conn.executemany(
            "INSERT INTO patients(patient_id, name, phone, phone_digits, address) "
            "VALUES (?,?,?,?,?)",
            [
                (
                    p.get("patient_id", ""),
                    p.get("name", ""),
                    str(p.get("phone") or ""),
                    _normalize_phone(str(p.get("phone") or "")),
                    p.get("address", ""),
                )
                for p in patients
            ],
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
    """Find patient by phone, matching with or without country code (e.g. +91)."""
    target = _normalize_phone(phone)
    if len(target) < MIN_PHONE_DIGITS:
        return None
    candidates = _phone_lookup_candidates(target)
    national = _phone_national_digits(target)
    with _db() as conn:
        placeholders = ",".join("?" * len(candidates))
        rows = conn.execute(
            "SELECT patient_id, name, phone, address, phone_digits FROM patients "
            f"WHERE phone_digits IN ({placeholders})",
            tuple(candidates),
        ).fetchall()
        # Fallback: stored value may use another prefix but share the last 10 digits
        if not rows and len(national) >= 10:
            rows = conn.execute(
                "SELECT patient_id, name, phone, address, phone_digits FROM patients "
                "WHERE length(phone_digits) >= 10 AND substr(phone_digits, -10) = ?",
                (national[-10:],),
            ).fetchall()
    if not rows:
        return None
    # Prefer longer stored form (+91…) over bare 10-digit duplicates, then stable id
    best = sorted(
        rows,
        key=lambda r: (-len(_normalize_phone(r["phone_digits"])), r["patient_id"]),
    )[0]
    return _patient_public(_row_to_dict(best))


def _find_patient_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return a patient only when exactly one row matches the name."""
    matches = _find_patients_by_name(name)
    if len(matches) == 1:
        return matches[0]
    return None


def _find_patients_by_name(name: str) -> List[Dict[str, Any]]:
    target = (name or "").strip().lower()
    if not target:
        return []
    with _db() as conn:
        rows = conn.execute(
            "SELECT patient_id, name, phone, address FROM patients "
            "WHERE lower(trim(name))=?",
            (target,),
        ).fetchall()
    return [_patient_public(_row_to_dict(r)) for r in rows]


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
        if ph and not _phones_match(ph, str(patient.get("phone", ""))):
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
        if not _phone_is_complete(ph):
            return {
                "ok": False,
                "patient": None,
                "message": (
                    "That phone number looks incomplete. Ask for the full 10-digit "
                    "mobile number (with or without +91), then look up again."
                ),
                "verified": False,
                "name_mismatch": False,
                "needs_phone": True,
                "incomplete_phone": True,
            }
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
            "message": (
                f"Existing patient found by phone: {patient.get('name')} "
                f"({patient.get('patient_id')}). "
                "Confirm this name aloud and get a yes before continuing."
            ),
            "verified": verified,
            "name_mismatch": False,
            "is_returning": True,
            "is_new": False,
            "needs_name": not bool(name),
            "confirm_name_from_db": patient.get("name"),
            "confirm_phone_from_db": patient.get("phone"),
        }

    if name and not require_name_with_phone:
        # Names are not unique — never treat name alone as identity.
        matches = _find_patients_by_name(name)
        if not matches:
            return {
                "ok": False,
                "message": (
                    "No patient found with that exact name. "
                    "Ask for the phone number first (names can be shared)."
                ),
                "verified": False,
                "name_mismatch": False,
                "needs_phone": True,
            }
        if len(matches) == 1:
            only = matches[0]
            return {
                "ok": False,
                "patient": only,
                "message": (
                    f"A patient named {only.get('name')} is on file with phone "
                    f"{only.get('phone')}. Names can be shared — ask them to confirm "
                    f"that phone number (or give their phone), then look up by phone."
                ),
                "verified": False,
                "name_mismatch": False,
                "is_returning": True,
                "needs_phone": True,
                "confirm_phone_from_db": only.get("phone"),
                "confirm_name_from_db": only.get("name"),
            }
        return {
            "ok": False,
            "message": (
                f"Several patients share the name '{name}'. "
                "Ask for the phone number — do not guess which person it is."
            ),
            "verified": False,
            "name_mismatch": False,
            "needs_phone": True,
            "name_match_count": len(matches),
        }

    return {
        "ok": False,
        "message": "Ask for the phone number first (required). Names alone are not unique.",
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
            conn.executemany(
                "INSERT INTO appointments("
                "appointment_id, patient_id, doctor_id, doctor, department, time, status"
                ") VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        a.get("appointment_id", ""),
                        a.get("patient_id", ""),
                        a.get("doctor_id", ""),
                        a.get("doctor", ""),
                        a.get("department", ""),
                        a.get("time", ""),
                        a.get("status", "BOOKED"),
                    )
                    for a in appts
                ],
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
        if not _appointment_status_is_live(str(a.get("status") or "")):
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
        if not _appointment_status_is_live(str(a.get("status") or "")):
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
    """Nearest free slots for this doctor around the requested time (same clinic hours).

    Accepts day-only phrases like 'today' / 'tomorrow' (searches remaining slots that day first).
    """
    now = now or datetime.now()
    preferred, day_only = parse_availability_anchor(preferred_time, now=now)
    if preferred is None:
        return []
    doc_id = (doctor_id or "").strip().upper()
    if not doc_id:
        return []

    taken = _active_booked_times_for_doctor(
        doc_id, exclude_appointment_id=exclude_appointment_id
    )
    preferred_key = preferred.strftime("%Y-%m-%d %H:%M")
    candidates: List[Tuple[timedelta, datetime]] = []
    for day_offset in range(0, max(1, int(search_days))):
        day = preferred.date() + timedelta(days=day_offset)
        for slot in _iter_clinic_slots_on_day(day, slot_minutes=slot_minutes):
            if slot <= now:
                continue
            key = slot.strftime("%Y-%m-%d %H:%M")
            if key in taken:
                continue
            # Exact clock requests skip the conflicted preferred slot; day-only includes it.
            if not day_only and key == preferred_key:
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
        if not _appointment_status_is_live(str(a.get("status") or "")):
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
    """Rank doctors: free at preferred time first, then fewer appointments that day.

    Day-only preferences ('today') rank by who still has open slots that day and
    attach nearest_times — they are not treated as fully unavailable.
    """
    req = normalize_appointment_time(preferred_time)
    day = resolve_appointment_day(preferred_time)
    if not req and not day:
        return list(doctors)

    day_prefix = (req[:10] if req else day.isoformat())
    day_loads = _appointment_counts_for_day(day_prefix)
    day_only = req is None
    ranked: List[Dict[str, Any]] = []
    for d in doctors:
        did = str(d.get("doctor_id") or "").strip().upper()
        row = dict(d)
        row["day_appointment_count"] = int(day_loads.get(did, 0))
        row["day"] = day_prefix
        if day_only:
            nearest = find_nearest_available_times(
                did,
                preferred_time,
                limit=3,
                search_days=1,
            )
            row["free_at_preferred_time"] = bool(nearest)
            row["nearest_times"] = nearest
            row["day_only_preference"] = True
        else:
            conflict = _find_conflict(
                None,
                str(d.get("name") or ""),
                req,
                doctor_id=did,
            )
            free_now = (
                conflict is None
                and not is_doctor_unavailable(did, req)
                and is_within_clinic_hours(req)
            )
            row["free_at_preferred_time"] = free_now
            row["day_only_preference"] = False
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
    *,
    now: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Return the patient's upcoming non-cancelled appointment, if any."""
    pid = (patient_id or "").strip()
    if not pid:
        return None

    rows: List[Dict[str, Any]]
    if appts is not None:
        rows = [a for a in appts if a.get("patient_id") == pid]
    else:
        with _db() as conn:
            fetched = conn.execute(
                "SELECT appointment_id, patient_id, doctor_id, doctor, department, time, status "
                "FROM appointments WHERE patient_id=? "
                "AND status NOT IN ('CANCELLED', 'COMPLETED') "
                "ORDER BY time ASC",
                (pid,),
            ).fetchall()
        rows = [_row_to_dict(r) for r in fetched]

    upcoming = [a for a in rows if _appointment_is_upcoming(a, now=now)]
    if not upcoming:
        return None
    upcoming.sort(key=lambda a: str(a.get("time") or ""))
    return upcoming[0]


def _complete_past_appointments_for_patient(
    patient_id: str,
    *,
    now: Optional[datetime] = None,
) -> List[str]:
    """Mark past BOOKED rows as COMPLETED (visit history kept). Returns appointment ids."""
    pid = (patient_id or "").strip()
    if not pid:
        return []
    completed: List[str] = []
    for appt in _load_all_appointments():
        if appt.get("patient_id") != pid:
            continue
        if str(appt.get("status") or "") != "BOOKED":
            continue
        if classify_appointment_timing(str(appt.get("time") or ""), now=now).get("timing_bucket") != "past":
            continue
        appt["status"] = "COMPLETED"
        _update_appointment(appt)
        aid = str(appt.get("appointment_id") or "")
        if aid:
            completed.append(aid)
    return completed


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


def build_department_booking_guidance(
    active_appointment: Optional[Dict[str, Any]],
    requested_department: str,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Decide what to tell the patient after identity + department are known.

    Actions:
      - inform_existing_soon: same dept, within a few hours — do not rebook
      - offer_prepone: same dept, days later — ask to move earlier
      - offer_new_booking: no same-dept appointment — ask to book that department
      - none: no requested department yet
    """
    req_dep = (requested_department or "").strip()
    if not req_dep:
        return {
            "action": "none",
            "same_department": None,
            "message": "Ask symptoms or department next.",
        }

    if not active_appointment:
        return {
            "action": "offer_new_booking",
            "same_department": False,
            "requested_department": req_dep,
            "message": (
                f"No active appointment for {req_dep}. "
                f"Ask: would you like to book a {req_dep} appointment?"
            ),
        }

    appt_dep = str(active_appointment.get("department") or "")
    same = department_matches(req_dep, appt_dep) or department_matches(appt_dep, req_dep)
    timing = classify_appointment_timing(str(active_appointment.get("time") or ""), now=now)
    bucket = timing.get("timing_bucket")
    aid = active_appointment.get("appointment_id")
    doctor = active_appointment.get("doctor")
    when = timing.get("appointment_time") or active_appointment.get("time")

    if same and bucket == "soon":
        return {
            "action": "inform_existing_soon",
            "same_department": True,
            "requested_department": req_dep,
            "timing": timing,
            "message": (
                f"Patient already has {req_dep} appointment {aid} with {doctor} at {when} "
                f"(within about {APPOINTMENT_SOON_HOURS} hours). "
                "Tell them they already have an appointment — do NOT book another or reschedule "
                "unless they explicitly ask to cancel or change it."
            ),
        }

    if same and bucket in ("later", "unknown"):
        return {
            "action": "offer_prepone",
            "same_department": True,
            "requested_department": req_dep,
            "timing": timing,
            "message": (
                f"Patient already has {req_dep} appointment {aid} with {doctor} at {when} "
                "(more than a few hours away). "
                "Ask: do you want to prepone (move earlier) this appointment? "
                "If yes, collect a sooner day/time and call reschedule_appointment."
            ),
        }

    if same and bucket == "past":
        return {
            "action": "offer_new_booking",
            "same_department": True,
            "requested_department": req_dep,
            "timing": timing,
            "message": (
                f"Last {req_dep} visit was {aid} with {doctor} at {when} (already passed). "
                "No need to cancel — book a new appointment directly if they want one."
            ),
        }

    # Active appointment exists but for a different department.
    return {
        "action": "offer_new_booking",
        "same_department": False,
        "requested_department": req_dep,
        "other_appointment": {
            "appointment_id": aid,
            "department": appt_dep,
            "doctor": doctor,
            "time": when,
        },
        "timing": timing,
        "message": (
            f"Patient has an appointment in {appt_dep or 'another department'} "
            f"({aid} with {doctor} at {when}), not {req_dep}. "
            f"Ask: would you like to book a {req_dep} appointment? "
            "If yes, explain they can only keep one active visit — confirm cancel of the other "
            "first, then book the new department."
        ),
    }


def _patient_ids_with_active_appointment(
    appts: Optional[List[Dict[str, Any]]] = None,
) -> set:
    rows = appts if appts is not None else _load_all_appointments()
    return {
        str(a.get("patient_id"))
        for a in rows
        if a.get("patient_id") and _appointment_is_upcoming(a)
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
