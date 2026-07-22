"""SQLite connection, schema, paths, and shared helpers."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import time as dt_time
from typing import Any, Dict, Iterator, List

BASE_DIR = os.path.dirname(__file__)


DATASET_DIR = os.path.join(BASE_DIR, "dataset")


DB_PATH = os.path.join(DATASET_DIR, "hospital.db")


DOCTORS_DB = os.path.join(DATASET_DIR, "doctors.json")


PATIENTS_DB = os.path.join(DATASET_DIR, "patients.json")


APPOINTMENTS_DB = os.path.join(DATASET_DIR, "appointments.json")


PRESCRIPTIONS_DB = os.path.join(DATASET_DIR, "prescriptions.json")


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


DEFAULT_LUNCH_START = "14:00"


DEFAULT_LUNCH_END = "15:00"


_CLINIC_START = dt_time(9, 0)


_CLINIC_END = dt_time(17, 0)


_DEFAULT_SLOT_MINUTES = 10


_initialized = False


_appointment_write_lock = threading.Lock()


APPOINTMENT_SOON_HOURS = 12


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


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
    from db_doctors import ensure_default_lunch_breaks
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


def _migrate_json_into(conn: sqlite3.Connection) -> None:
    """Import dataset/*.json into SQLite when present (one-time)."""
    from db_patients import _normalize_phone
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


def clear_table(table: str) -> int:
    """Delete all rows from a table. Returns previous row count."""
    allowed = {"doctors", "patients", "appointments", "prescriptions"}
    if table not in allowed:
        raise ValueError(f"Unknown table: {table}")
    with _db() as conn:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(f"DELETE FROM {table}")
    return int(count)
