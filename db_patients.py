"""Patient records, phone matching, and identity resolution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db_core import _db, _next_id, _row_to_dict

def _normalize_phone(phone: str) -> str:
    return "".join(ch for ch in (phone or "") if ch.isdigit())


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
    from db_appointments import _load_all_appointments
    from db_doctors import _get_doctor_by_id, _get_doctor_by_name
    from db_prescriptions import _load_prescriptions
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
