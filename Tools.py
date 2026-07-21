import json
import logging
import sqlite3
from typing import Optional
from langchain_core.tools import tool
from database import (
    _load_doctors,
    _get_doctor_by_name,
    _get_doctor_by_id,
    _load_all_appointments,
    _find_conflict,
    _get_next_appointment_id,
    _insert_appointment,
    _update_appointment,
    _get_or_create_patient,
    _enrich_appointment,
    _find_prescriptions,
    _resolve_patient,
    _patient_past_doctors,
    _names_match,
    _find_patient_by_phone,
    _find_active_appointment_for_patient,
    _complete_past_appointments_for_patient,
    _normalize_phone,
    _phone_is_complete,
    build_availability_suggestions,
    build_department_booking_guidance,
    check_slot_bookable,
    classify_appointment_timing,
    department_matches,
    find_nearest_available_times,
    is_within_clinic_hours,
    normalize_appointment_time,
    rank_doctors_for_preferred_time,
    resolve_appointment_day,
)

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def _conflict_payload(
    doctor: dict,
    preferred_time: str,
    conflict: Optional[dict] = None,
    *,
    exclude_appointment_id: str = "",
    reason: str = "time_conflict",
    message: str = "",
) -> dict:
    """Refuse the slot and attach nearest times + alternate doctors. Never overwrites."""
    suggestions = build_availability_suggestions(
        doctor,
        preferred_time,
        exclude_appointment_id=exclude_appointment_id,
    )
    payload = {
        "ok": False,
        "reason": reason or "time_conflict",
        "message": message or suggestions["message"],
        "conflict": _enrich_appointment(conflict) if conflict else None,
        "suggestions": suggestions,
        "nearest_times": suggestions.get("nearest_times") or [],
        "alternate_doctors": suggestions.get("alternate_doctors") or [],
    }
    return payload


def _slot_refusal(doctor: dict, preferred_time: str, refusal: dict, *, exclude_id: str = "") -> str:
    """Build JSON for outside-hours / unavailable / conflict refusals with suggestions."""
    reason = refusal.get("reason") or "time_conflict"
    conflict = refusal.get("conflict")
    base_msg = refusal.get("message") or ""
    if reason in ("time_conflict", "doctor_unavailable", "outside_clinic_hours"):
        payload = _conflict_payload(
            doctor,
            preferred_time,
            conflict,
            exclude_appointment_id=exclude_id,
            reason=reason,
            message=base_msg,
        )
        if reason == "outside_clinic_hours":
            payload["message"] = (
                f"{base_msg} Offer a time between 9:00 AM and 5:00 PM. "
                f"Nearest open options: {', '.join(payload.get('nearest_times') or []) or 'none found'}."
            )
        elif reason == "doctor_unavailable":
            payload["unavailable"] = refusal.get("unavailable")
            payload["message"] = (
                f"{base_msg} "
                f"Nearest open times: {', '.join(payload.get('nearest_times') or []) or 'none'}. "
                "Or offer an alternate doctor free at the requested time."
            )
        return json.dumps(payload)
    return json.dumps(refusal)


@tool
def lookup_patient(
    phone: Optional[str] = None,
    patient_name: Optional[str] = None,
    patient_id: Optional[str] = None,
    department: Optional[str] = None,
) -> str:
    """Look up a patient. Phone is the primary key — always ask phone first.

    Names are not unique. Never identify someone from name alone or from chat memory.
    After phone lookup: if returning, confirm the name returned in confirm_name_from_db;
    if new, ask for their full name then save_patient.
    """
    logging.info(
        "lookup_patient id=%s phone=%s name=%s department=%s",
        patient_id,
        phone,
        patient_name,
        department,
    )

    phone_s = (phone or "").strip()
    name_s = (patient_name or "").strip()
    pid_s = (patient_id or "").strip()

    if not any([pid_s, phone_s, name_s]):
        return json.dumps({
            "ok": False,
            "message": "Ask for the phone number first. Names can be the same for different people.",
            "needs_phone": True,
        })

    # Name without phone: do not treat as identified — resolve will force needs_phone.
    resolved = _resolve_patient(
        patient_id=pid_s or None,
        phone=phone_s or None,
        patient_name=name_s or None,
        require_name_with_phone=False,
    )

    def _confirm_fields(src: dict) -> dict:
        out = {}
        if src.get("confirm_name_from_db"):
            out["confirm_name_from_db"] = src.get("confirm_name_from_db")
        if src.get("confirm_phone_from_db"):
            out["confirm_phone_from_db"] = src.get("confirm_phone_from_db")
        if src.get("name_match_count") is not None:
            out["name_match_count"] = src.get("name_match_count")
        return out

    if resolved.get("name_mismatch"):
        payload = {
            "ok": False,
            "message": resolved.get("message"),
            "name_mismatch": True,
            "patient": resolved.get("patient"),
            "is_returning": True,
            "needs_phone": not bool(phone_s),
            "needs_name": True,
        }
        payload.update(_confirm_fields(resolved))
        if resolved.get("patient"):
            payload["confirm_name_from_db"] = (resolved.get("patient") or {}).get("name")
            payload["confirm_phone_from_db"] = (resolved.get("patient") or {}).get("phone")
        return json.dumps(payload)

    if resolved.get("is_new"):
        return json.dumps({
            "ok": True,
            "is_returning": False,
            "is_new": True,
            "message": (
                "No patient on file for this phone. Ask for their full name, confirm it, "
                "then call save_patient(patient_name=..., phone=...)."
            ),
            "needs_name": True,
            "needs_phone": False,
        })

    if not resolved.get("ok"):
        payload = {
            "ok": False,
            "message": resolved.get("message"),
            "needs_phone": resolved.get("needs_phone", False),
            "needs_name": resolved.get("needs_name", False),
            "patient": resolved.get("patient"),
        }
        if resolved.get("incomplete_phone"):
            payload["incomplete_phone"] = True
        payload.update(_confirm_fields(resolved))
        return json.dumps(payload)

    patient = resolved.get("patient")
    if not patient:
        return json.dumps({
            "ok": False,
            "message": resolved.get("message") or "Patient not found.",
            "needs_phone": True,
        })

    past = _patient_past_doctors(str(patient.get("patient_id")))
    active = _find_active_appointment_for_patient(str(patient.get("patient_id")))
    active_enriched = _enrich_appointment(active) if active else None
    timing = None
    if active_enriched:
        timing = classify_appointment_timing(str(active_enriched.get("time") or ""))
        active_enriched = dict(active_enriched)
        active_enriched["timing"] = timing

    confirm_name = resolved.get("confirm_name_from_db") or patient.get("name")
    confirm_phone = resolved.get("confirm_phone_from_db") or patient.get("phone")
    name_confirmed = bool(name_s) and _names_match(name_s, str(confirm_name or ""))

    if not name_confirmed:
        msg = (
            f"Phone matches patient {patient.get('patient_id')}. "
            f"Confirm name from database: ask 'Are you {confirm_name}?' and wait for yes. "
            f"Then call lookup_patient(phone=..., patient_name={confirm_name!r}) "
            "before discussing appointments. "
            "Do NOT say there is no appointment while active_appointment is present below."
        )
    elif active_enriched:
        msg = (
            f"Patient verified as {confirm_name}. Active appointment "
            f"{active_enriched.get('appointment_id')} with {active_enriched.get('doctor')} "
            f"at {active_enriched.get('time')} "
            f"(department {active_enriched.get('department') or 'unknown'}, "
            f"timing {timing.get('timing_bucket')})."
        )
    else:
        msg = (
            f"Patient verified as {confirm_name} ({patient.get('patient_id')}). "
            "No active appointment on file."
        )

    guidance = None
    dep = (department or "").strip()
    # Guidance only after name confirm so we do not push booking before identity.
    if dep and name_confirmed:
        guidance = build_department_booking_guidance(active_enriched, dep)
        if guidance.get("message"):
            msg = f"{msg} {guidance['message']}"

    try:
        from conversation_log import get_current_call_id, link_call_patient

        call_id = get_current_call_id()
        if call_id and name_confirmed:
            link_call_patient(
                call_id,
                patient_id=str(patient.get("patient_id") or ""),
                phone=str(patient.get("phone") or phone or ""),
                appointment_id=str((active_enriched or {}).get("appointment_id") or ""),
            )
    except Exception:
        pass

    payload = {
        "ok": True,
        "is_returning": True,
        "is_new": False,
        "verified": name_confirmed,
        "needs_name": not name_confirmed,
        "needs_phone": False,
        "message": msg,
        "patient": patient,
        # Always include appointment when found. Hiding it until name confirm made the
        # model say "no appointments" after the user said yes but the agent re-called
        # lookup without patient_name (active_appointment was null).
        "past_doctors": past,
        "last_doctor": (past[0] if past else None),
        "active_appointment": active_enriched,
        "confirm_name_from_db": confirm_name,
        "confirm_phone_from_db": confirm_phone,
    }
    if guidance:
        payload["booking_guidance"] = guidance
        payload["requested_department"] = dep
    return json.dumps(payload)


@tool
def save_patient(
    patient_name: str,
    phone: str,
    address: str = "",
) -> str:
    """Create or update a patient record after phone and name are confirmed.

    Call this ONLY after the user has confirmed both phone and full name.
    This writes the patient to the patients table. Booking still needs book_appointment later.

    Args:
        patient_name: Confirmed full name.
        phone: Confirmed phone number.
        address: Optional address.

    Returns:
        JSON with ok, patient (including patient_id), and whether created or updated.
    """
    logging.info("save_patient %s %s", patient_name, phone)
    name = (patient_name or "").strip()
    ph = (phone or "").strip()
    if not name:
        return json.dumps({"ok": False, "message": "patient_name is required."})
    if not ph:
        return json.dumps({"ok": False, "message": "phone is required."})
    if not _phone_is_complete(ph):
        return json.dumps({
            "ok": False,
            "message": (
                f"Phone '{ph}' looks incomplete ({len(_normalize_phone(ph))} digits). "
                "Ask for the full 10-digit mobile number before saving."
            ),
            "needs_phone": True,
            "incomplete_phone": True,
        })

    existing = _find_patient_by_phone(ph)
    if existing is not None and not _names_match(name, str(existing.get("name", ""))):
        return json.dumps({
            "ok": False,
            "message": (
                f"Phone is already registered to {existing.get('name')} "
                f"({existing.get('patient_id')}). Confirm the correct name."
            ),
            "name_mismatch": True,
            "patient": existing,
        })

    before_id = existing.get("patient_id") if existing else None
    patient = _get_or_create_patient(name=name, phone=ph, address=(address or "").strip())
    created = before_id is None
    try:
        from conversation_log import get_current_call_id, link_call_patient

        call_id = get_current_call_id()
        if call_id and patient:
            link_call_patient(
                call_id,
                patient_id=str(patient.get("patient_id") or ""),
                phone=str(patient.get("phone") or ph),
            )
    except Exception:
        pass
    return json.dumps({
        "ok": True,
        "message": "Patient saved" if created else "Patient updated",
        "created": created,
        "patient": patient,
    })


@tool
def book_appointment(
    patient_name: str,
    doctor: str,
    time: str,
    phone: str = "",
    address: str = "",
) -> str:
    """Book an appointment after validating doctor exists and time is free.

    Reuses an existing patient when phone matches; creates a new patient otherwise.
    Call lookup_patient first when you have a phone so returning patients can reuse a past doctor.
    Past visits do not block a new booking (they are archived as COMPLETED automatically).
    Only one upcoming appointment at a time — cancel or reschedule that first if it is still in the future.
    If the doctor is already booked at that time, returns ok=false with nearest_times and
    alternate_doctors — never deletes or replaces another patient's appointment.

    Args:
        patient_name: The full name of the patient.
        doctor: The doctor's name the patient wants to see.
        time: Appointment time. Prefer phrases like 'tomorrow 10 AM' or ISO; they are
            stored as a concrete local datetime (YYYY-MM-DD HH:MM).
        phone: Patient phone number (required to match existing records).
        address: Patient address (optional; stored on new patients).

    Returns:
        JSON string with result: ok, message, patient, and appointment if created.
        On conflict: nearest_times + alternate_doctors for the agent to offer aloud.
    """
    logging.info("book_appointment %s %s %s %s", patient_name, phone, doctor, time)
    doctor_row = _get_doctor_by_name(doctor)
    if not doctor_row:
        return json.dumps({
            "ok": False,
            "message": f"Doctor not found: {doctor}. Use the full name as in the directory.",
        })

    if not (patient_name or "").strip():
        return json.dumps({"ok": False, "message": "patient_name is required."})
    if not (phone or "").strip():
        return json.dumps({"ok": False, "message": "phone is required so we can find or create the patient."})

    normalized_time = normalize_appointment_time(time)
    if not normalized_time:
        return json.dumps({
            "ok": False,
            "message": (
                f"Could not understand the appointment time '{time}'. "
                "Please give a day and clock time, for example 'tomorrow at 10 AM' "
                "or '2026-07-18 10:00'."
            ),
        })

    existing = _find_patient_by_phone(phone)
    if existing is not None and not _names_match(patient_name, str(existing.get("name", ""))):
        return json.dumps({
            "ok": False,
            "message": (
                f"Phone is already registered to {existing.get('name')} "
                f"({existing.get('patient_id')}). Confirm the correct name before booking."
            ),
            "name_mismatch": True,
            "patient": existing,
        })

    patient = _get_or_create_patient(
        name=patient_name.strip(),
        phone=(phone or "").strip(),
        address=(address or "").strip(),
    )

    # Past visits stay in the record as COMPLETED; they do not block a new booking.
    archived = _complete_past_appointments_for_patient(patient["patient_id"])

    appts = _load_all_appointments()
    existing_appt = _find_active_appointment_for_patient(patient["patient_id"], appts)
    if existing_appt is not None:
        existing_enriched = _enrich_appointment(existing_appt)
        guidance = build_department_booking_guidance(
            existing_enriched,
            str(doctor_row.get("department") or ""),
        )
        action = guidance.get("action")
        if action == "inform_existing_soon":
            detail = (
                f"{patient.get('name')} already has an appointment soon "
                f"({existing_appt.get('appointment_id')}) with {existing_appt.get('doctor')} "
                f"at {existing_appt.get('time')}. Tell them they already have an appointment — "
                "do not book another."
            )
        elif action == "offer_prepone":
            detail = (
                f"{patient.get('name')} already has an appointment "
                f"({existing_appt.get('appointment_id')}) with {existing_appt.get('doctor')} "
                f"at {existing_appt.get('time')}. Ask if they want to prepone it, then use "
                "reschedule_appointment — do not create a second booking."
            )
        elif action == "offer_new_booking":
            detail = (
                f"{patient.get('name')} has another upcoming visit "
                f"({existing_appt.get('appointment_id')}) in a different department. "
                "Confirm cancel or reschedule that one before booking this department."
            )
        else:
            detail = (
                f"{patient.get('name')} already has an upcoming appointment "
                f"({existing_appt.get('appointment_id')}) with {existing_appt.get('doctor')} "
                f"at {existing_appt.get('time')}. Cancel or reschedule that one first — "
                "a patient may have only one upcoming visit at a time."
            )
        return json.dumps({
            "ok": False,
            "message": detail,
            "existing_appointment": existing_enriched,
            "booking_guidance": guidance,
            "patient": patient,
            "archived_past_visits": archived,
        })

    refusal = check_slot_bookable(
        str(doctor_row.get("doctor_id") or ""),
        normalized_time,
        doctor_name=doctor_row["name"],
    )
    if refusal:
        return _slot_refusal(doctor_row, normalized_time, refusal)

    appointment_id = _get_next_appointment_id()
    new_appt = {
        "appointment_id": appointment_id,
        "patient_id": patient["patient_id"],
        "doctor_id": doctor_row.get("doctor_id"),
        "doctor": doctor_row["name"],
        "department": doctor_row.get("department", ""),
        "time": normalized_time,
        "status": "BOOKED",
    }
    try:
        _insert_appointment(new_appt)
    except sqlite3.IntegrityError:
        refusal = check_slot_bookable(
            str(doctor_row.get("doctor_id") or ""),
            normalized_time,
            doctor_name=doctor_row["name"],
        ) or {
            "reason": "time_conflict",
            "message": f"Time conflict at '{normalized_time}'.",
            "conflict": _find_conflict(
                None,
                doctor_row["name"],
                normalized_time,
                doctor_id=str(doctor_row.get("doctor_id") or ""),
            ),
        }
        return _slot_refusal(doctor_row, normalized_time, refusal)

    try:
        from conversation_log import record_booking_chat

        record_booking_chat(patient=patient, appointment=new_appt)
    except Exception:
        logging.exception("Failed to record booking chat for %s", patient.get("patient_id"))

    return json.dumps({
        "ok": True,
        "message": "Appointment booked",
        "patient": patient,
        "appointment": _enrich_appointment(new_appt),
        "time_resolved_from": time,
        "archived_past_visits": archived,
    })


@tool
def cancel_appointment(
    appointment_id: str = "",
    phone: str = "",
) -> str:
    """Cancel an existing appointment by ID, or by phone if the patient has one active appointment.

    Args:
        appointment_id: Appointment id like APT-0001 when known.
        phone: Patient phone — used to find their active appointment when id is unknown.

    Returns:
        JSON string indicating whether the cancellation succeeded and the record (if any).
    """
    logging.info("cancel_appointment id=%s phone=%s", appointment_id, phone)

    appts = _load_all_appointments()
    target = None
    aid = (appointment_id or "").strip().upper()
    if aid:
        for a in appts:
            if str(a.get("appointment_id", "")).upper() == aid:
                target = a
                break
    elif (phone or "").strip():
        patient = _find_patient_by_phone(phone)
        if not patient:
            return json.dumps({"ok": False, "message": "No patient found for that phone."})
        target = _find_active_appointment_for_patient(str(patient.get("patient_id")), appts)
        if target is None:
            return json.dumps({
                "ok": False,
                "message": f"No active appointment found for {patient.get('name')}.",
                "patient": patient,
            })
    else:
        return json.dumps({
            "ok": False,
            "message": "Provide appointment_id or phone to cancel.",
        })

    if target is None:
        return json.dumps({"ok": False, "message": f"Appointment not found: {appointment_id}"})

    if target.get("status") == "CANCELLED":
        return json.dumps({
            "ok": True,
            "message": "Appointment already cancelled",
            "appointment": _enrich_appointment(target),
        })

    target["status"] = "CANCELLED"
    _update_appointment(target)
    return json.dumps({
        "ok": True,
        "message": "Appointment cancelled",
        "appointment": _enrich_appointment(target),
    })


@tool
def reschedule_appointment(
    appointment_id: str = "",
    new_time: str = "",
    phone: str = "",
) -> str:
    """Reschedule an existing appointment to a new time.

    Prefer appointment_id when known. If unknown, pass phone to find the patient's
    one active appointment. new_time can be 'tomorrow 4:50 PM' and is stored as YYYY-MM-DD HH:MM.
    If the doctor is already booked at the new time, returns nearest_times and
    alternate_doctors — never deletes another patient's appointment.

    Args:
        appointment_id: ID like APT-0001 when known.
        new_time: Target day/time (free-text or ISO).
        phone: Patient phone when appointment id is unknown.

    Returns:
        JSON string with ok, message, and updated appointment if successful.
        On conflict: nearest_times + alternate_doctors to offer the patient.
    """
    logging.info(
        "reschedule_appointment id=%s phone=%s new_time=%s",
        appointment_id,
        phone,
        new_time,
    )
    if not (new_time or "").strip():
        return json.dumps({"ok": False, "message": "new_time is required."})

    appts = _load_all_appointments()
    target = None
    aid = (appointment_id or "").strip().upper()
    if aid:
        for a in appts:
            if str(a.get("appointment_id", "")).upper() == aid:
                target = a
                break
        if target is None:
            return json.dumps({"ok": False, "message": f"Appointment not found: {appointment_id}"})
    elif (phone or "").strip():
        patient = _find_patient_by_phone(phone)
        if not patient:
            return json.dumps({"ok": False, "message": "No patient found for that phone."})
        target = _find_active_appointment_for_patient(str(patient.get("patient_id")), appts)
        if target is None:
            return json.dumps({
                "ok": False,
                "message": f"No active appointment found for {patient.get('name')}.",
                "patient": patient,
            })
    else:
        return json.dumps({
            "ok": False,
            "message": "Provide appointment_id or phone, plus the new_time.",
        })

    if target.get("status") == "CANCELLED":
        return json.dumps({"ok": False, "message": "Cannot reschedule a cancelled appointment."})

    normalized_time = normalize_appointment_time(new_time)
    if not normalized_time:
        return json.dumps({
            "ok": False,
            "message": (
                f"Could not understand the new time '{new_time}'. "
                "Please give a day and clock time, for example 'tomorrow at 10 AM'."
            ),
        })

    doctor_name = target.get("doctor", "")
    doctor_id = str(target.get("doctor_id") or "")
    doctor_row = {
        "doctor_id": doctor_id,
        "name": doctor_name,
        "department": target.get("department", ""),
    }
    if doctor_id:
        known = _get_doctor_by_name(doctor_name) if doctor_name else None
        if known is None:
            known = _get_doctor_by_id(doctor_id)
        if known:
            doctor_row = known

    exclude_id = str(target.get("appointment_id") or "")
    refusal = check_slot_bookable(
        doctor_id,
        normalized_time,
        exclude_appointment_id=exclude_id,
        doctor_name=doctor_name,
    )
    if refusal:
        return _slot_refusal(doctor_row, normalized_time, refusal, exclude_id=exclude_id)

    target["time"] = normalized_time
    try:
        _update_appointment(target)
    except sqlite3.IntegrityError:
        refusal = check_slot_bookable(
            doctor_id,
            normalized_time,
            exclude_appointment_id=exclude_id,
            doctor_name=doctor_name,
        ) or {
            "reason": "time_conflict",
            "message": f"Time conflict at '{normalized_time}'.",
        }
        return _slot_refusal(doctor_row, normalized_time, refusal, exclude_id=exclude_id)
    return json.dumps({
        "ok": True,
        "message": "Appointment rescheduled",
        "appointment": _enrich_appointment(target),
        "time_resolved_from": new_time,
    })


@tool
def list_doctors(
    department: Optional[str] = None,
    query: Optional[str] = None,
    exclude_doctor: Optional[str] = None,
    preferred_time: Optional[str] = None,
    limit: int = 10,
) -> str:
    """List doctors from the directory, optionally filtered by department or a name query.

    For a second opinion in the same department, pass department=last department and
    exclude_doctor=previous doctor name so other doctors in that department are returned.
    When preferred_time is known, pass it so results prefer doctors free at that time
    and with fewer appointments that day. If nobody is free at that exact time (e.g. lunch
    2:00–3:00 PM), still returns doctors in the department with nearest_times — do NOT
    switch departments.

    Args:
        department: Filter by department (aliases ok: 'Dental' → Dentistry).
        query: Filter by name (case-insensitive; matches without the 'Dr.' prefix).
        exclude_doctor: Doctor name to omit (e.g. the patient's last doctor).
        preferred_time: Optional day/time — ranks free-at-that-time and least-busy first.
        limit: Maximum number of results to return.

    Returns:
        JSON with ok, count, doctors. When preferred time is blocked for everyone,
        preferred_time_unavailable=true and each doctor may include nearest_times.
    """
    logging.info(
        "list_doctors dep=%s query=%s exclude=%s preferred_time=%s",
        department,
        query,
        exclude_doctor,
        preferred_time,
    )
    docs = _load_doctors()
    dep = (department or "").strip()
    q = (query or "").strip().lower()

    def _norm_name(n: str) -> str:
        n = n.strip().lower()
        if n.startswith("dr. "):
            n = n[4:]
        elif n.startswith("dr "):
            n = n[3:]
        return n

    if dep:
        docs = [d for d in docs if department_matches(dep, d.get("department", ""))]

    if q:
        docs = [d for d in docs if q in _norm_name(d.get("name", ""))]

    if exclude_doctor:
        ex = _norm_name(exclude_doctor)
        docs = [d for d in docs if _norm_name(d.get("name", "")) != ex]

    preferred = (preferred_time or "").strip()
    preferred_norm = normalize_appointment_time(preferred) if preferred else None
    preferred_day = resolve_appointment_day(preferred) if preferred else None
    preferred_unavailable = False
    message = ""
    day_only = bool(preferred and preferred_day and not preferred_norm)

    if preferred and (preferred_norm or preferred_day):
        docs = rank_doctors_for_preferred_time(docs, preferred)
        free_only = [d for d in docs if d.get("free_at_preferred_time")]
        if free_only:
            docs = free_only
            if day_only:
                # Patient named a day but no clock — offer concrete open times.
                enriched = []
                sample = []
                for d in docs[: max(1, int(limit))]:
                    row = dict(d)
                    nearest = row.get("nearest_times") or find_nearest_available_times(
                        str(d.get("doctor_id") or ""),
                        preferred,
                        limit=3,
                        search_days=1,
                    )
                    row["nearest_times"] = nearest
                    enriched.append(row)
                    if nearest:
                        sample.append(f"{d.get('name')} at {nearest[0]}")
                docs = enriched
                message = (
                    f"Patient asked for {preferred_day.isoformat()} without a clock time. "
                    f"Doctors ARE available that day — offer nearest_times "
                    + (f"(e.g. {'; '.join(sample)})" if sample else "")
                    + " and ask which time they want. Do NOT say the doctor/day is unavailable."
                )
        elif docs:
            # Exact clock blocked for all, OR day has no remaining slots.
            preferred_unavailable = True
            outside = preferred_norm and not is_within_clinic_hours(preferred_norm)
            if day_only:
                reason_bit = "has no remaining open slots"
                label = preferred_day.isoformat() if preferred_day else preferred
            elif outside:
                reason_bit = "outside clinic hours (9:00 AM–5:00 PM)"
                label = preferred_norm or preferred
            else:
                reason_bit = "not available (often lunch 2:00–3:00 PM, or fully booked)"
                label = preferred_norm or preferred
            enriched = []
            for d in docs[: max(1, int(limit))]:
                row = dict(d)
                nearest = find_nearest_available_times(
                    str(d.get("doctor_id") or ""),
                    preferred_norm or preferred,
                    limit=3,
                    search_days=3 if day_only else 3,
                )
                row["nearest_times"] = nearest
                enriched.append(row)
            docs = enriched
            sample = []
            for d in docs[:3]:
                nt = d.get("nearest_times") or []
                if nt:
                    sample.append(f"{d.get('name')} at {nt[0]}")
            message = (
                f"Doctors are in this department, but {label} is {reason_bit}. "
                f"Offer nearest times in the SAME department"
                + (f" (e.g. {'; '.join(sample)})" if sample else "")
                + ". Do NOT switch to another department."
            )
    elif preferred:
        # Unparseable preferred_time — ignore for availability; do not claim unavailable.
        message = (
            f"Could not parse preferred_time '{preferred}' as a day/time. "
            "Ask for a specific day and clock time (clinic 9:00 AM–5:00 PM)."
        )

    docs = docs[: max(1, int(limit))]
    payload = {
        "ok": True,
        "count": len(docs),
        "doctors": docs,
        "department_filter": dep or None,
        "preferred_time": preferred_norm or (preferred_day.isoformat() if preferred_day else None) or preferred or None,
        "preferred_time_unavailable": preferred_unavailable,
        "day_only_preference": day_only,
    }
    if message:
        payload["message"] = message
    if dep and not docs:
        payload["message"] = (
            f"No doctors matched department '{dep}'. "
            "Only then may you try a closely related department — not because of a busy time."
        )
    return json.dumps(payload)


@tool
def get_prescriptions(
    patient_name: Optional[str] = None,
    phone: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> str:
    """Look up medicines prescribed for a patient.

    If patient_id is known, use it. Otherwise verify with BOTH name and phone
    (patients often forget their id). Collect missing fields one turn at a time.

    Args:
        patient_name: Full patient name (required with phone when id unknown).
        phone: Patient phone number (required with name when id unknown).
        patient_id: Patient id like PAT-0001 if known.

    Returns:
        JSON with prescriptions: medicine_name, timing, doctor, prescription_id.
    """
    logging.info("get_prescriptions patient_id=%s phone=%s name=%s", patient_id, phone, patient_name)

    if not any([(patient_id or "").strip(), (phone or "").strip(), (patient_name or "").strip()]):
        return json.dumps({
            "ok": False,
            "message": "Ask for patient id if they know it, otherwise start with their phone number.",
            "needs_phone": True,
        })

    result = _find_prescriptions(
        patient_id=(patient_id or "").strip() or None,
        patient_name=(patient_name or "").strip() or None,
        phone=(phone or "").strip() or None,
    )
    return json.dumps(result)
