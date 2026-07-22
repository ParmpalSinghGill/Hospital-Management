"""Appointments, conflicts, availability, and booking guidance."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from db_core import (
    APPOINTMENT_SOON_HOURS,
    _DEFAULT_SLOT_MINUTES,
    _appointment_status_is_live,
    _appointment_write_lock,
    _db,
    _next_id,
    _row_to_dict,
)
from db_doctors import (
    _get_doctor_by_id,
    _get_doctor_by_name,
    _load_doctors,
    _normalize_doctor_name,
    department_matches,
    is_doctor_unavailable,
)
from db_patients import _get_patient_by_id
from db_time import (
    _iter_clinic_slots_on_day,
    classify_appointment_timing,
    is_within_clinic_hours,
    normalize_appointment_time,
    parse_appointment_datetime,
    parse_availability_anchor,
    resolve_appointment_day,
)

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
