import json
import logging
import sqlite3

from langchain_core.tools import tool

from audit_log import audit, subject_ref
from phi_redact import mask_phone

from database import (
    _complete_past_appointments_for_patient,
    _enrich_appointment,
    _find_active_appointment_for_patient,
    _find_conflict,
    _find_patient_by_phone,
    _get_doctor_by_id,
    _get_doctor_by_name,
    _get_next_appointment_id,
    _get_or_create_patient,
    _insert_appointment,
    _load_all_appointments,
    _names_match,
    _update_appointment,
    build_department_booking_guidance,
    check_slot_bookable,
    normalize_appointment_time,
)
from tools_common import _slot_refusal


@tool
def book_appointment(
    patient_name: str,
    doctor: str,
    time: str,
    phone: str = "",
    address: str = "",
) -> str:
    """Book an appointment after validating doctor exists and time is free."""
    logging.info("book_appointment phone=%s doctor=%s time=%s", mask_phone(phone), doctor, time)
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
        audit("tool.book_appointment", actor="agent", outcome="name_mismatch", subject=subject_ref(phone=phone))
        return json.dumps({
            "ok": False,
            "message": (
                "The name given does not match our record for this phone number. Ask the caller "
                "to repeat their full name as registered; do not suggest names or book."
            ),
            "name_mismatch": True,
            "needs_name": True,
        })

    patient = _get_or_create_patient(
        name=patient_name.strip(),
        phone=(phone or "").strip(),
        address=(address or "").strip(),
    )

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
        from conversation_log import get_current_call_id, link_call_patient, record_booking_chat

        # Prefer tagging the live call that just booked — don't invent a fake transcript.
        live_id = get_current_call_id()
        if live_id:
            link_call_patient(
                live_id,
                patient_id=str(patient.get("patient_id") or ""),
                phone=str(patient.get("phone") or ""),
                appointment_id=str(new_appt.get("appointment_id") or ""),
            )
        else:
            record_booking_chat(patient=patient, appointment=new_appt)
    except Exception:
        logging.exception("Failed to record booking chat for %s", patient.get("patient_id"))

    _audit_booking(patient, new_appt)
    return json.dumps({
        "ok": True,
        "message": "Appointment booked",
        "patient": patient,
        "appointment": _enrich_appointment(new_appt),
        "time_resolved_from": time,
        "archived_past_visits": archived,
    })


def _audit_booking(patient: dict, appointment: dict) -> None:
    audit("tool.book_appointment", actor="agent",
          subject=subject_ref(patient_id=str((patient or {}).get("patient_id") or "")),
          detail={"appointment_id": str((appointment or {}).get("appointment_id") or "")})


@tool
def cancel_appointment(
    appointment_id: str = "",
    phone: str = "",
    patient_name: str = "",
) -> str:
    """Cancel an appointment. Always pass the caller's phone (and full name when no
    appointment_id is known). The phone must belong to the appointment's patient."""
    logging.info("cancel_appointment id=%s phone=%s", appointment_id, mask_phone(phone))

    target, err = _authorize_appointment(appointment_id, phone, patient_name, action="cancel")
    if err:
        return json.dumps(err)

    if target.get("status") == "CANCELLED":
        return json.dumps({
            "ok": True,
            "message": "Appointment already cancelled",
            "appointment": _enrich_appointment(target),
        })

    target["status"] = "CANCELLED"
    _update_appointment(target)
    audit("tool.cancel_appointment", actor="agent",
          subject=subject_ref(patient_id=str(target.get("patient_id") or "")),
          detail={"appointment_id": target.get("appointment_id")})
    return json.dumps({
        "ok": True,
        "message": "Appointment cancelled",
        "appointment": _enrich_appointment(target),
    })


def _authorize_appointment(appointment_id: str, phone: str, patient_name: str, *, action: str):
    """Resolve the target appointment and prove the caller owns it.

    Returns (appointment, None) or (None, error_payload).
    """
    aid = (appointment_id or "").strip().upper()
    ph = (phone or "").strip()
    name = (patient_name or "").strip()
    if not ph:
        return None, {
            "ok": False,
            "message": f"Ask for the caller's phone number before you {action} anything.",
            "needs_phone": True,
        }
    patient = _find_patient_by_phone(ph)
    if not patient:
        return None, {"ok": False, "message": "No patient found for that phone.", "needs_phone": True}
    if name and not _names_match(name, str(patient.get("name", ""))):
        audit(f"tool.{action}_appointment", actor="agent", outcome="name_mismatch", subject=subject_ref(phone=ph))
        return None, {
            "ok": False,
            "message": "The name given does not match our record for this phone. Ask them to repeat their full name.",
            "name_mismatch": True,
            "needs_name": True,
        }
    appts = _load_all_appointments()
    if aid:
        target = next((a for a in appts if str(a.get("appointment_id", "")).upper() == aid), None)
        if target is None:
            return None, {"ok": False, "message": f"Appointment not found: {appointment_id}"}
        if str(target.get("patient_id") or "") != str(patient.get("patient_id") or ""):
            audit(f"tool.{action}_appointment", actor="agent", outcome="not_owner",
                  subject=subject_ref(patient_id=str(patient.get("patient_id") or "")),
                  detail={"appointment_id": aid})
            return None, {
                "ok": False,
                "message": "That appointment is not registered to this phone number. Verify the details with the caller.",
                "not_owner": True,
            }
        return target, None
    if not name:
        return None, {
            "ok": False,
            "message": f"Ask for the caller's full name to verify before you {action} by phone.",
            "needs_name": True,
        }
    target = _find_active_appointment_for_patient(str(patient.get("patient_id")), appts)
    if target is None:
        return None, {
            "ok": False,
            "message": "No active appointment found for this patient.",
            "patient": patient,
        }
    return target, None


@tool
def reschedule_appointment(
    appointment_id: str = "",
    new_time: str = "",
    phone: str = "",
    patient_name: str = "",
) -> str:
    """Reschedule an appointment to a new time. Always pass the caller's phone (and full
    name when no appointment_id is known); the phone must belong to the appointment."""
    logging.info("reschedule_appointment id=%s phone=%s new_time=%s", appointment_id, mask_phone(phone), new_time)
    if not (new_time or "").strip():
        return json.dumps({"ok": False, "message": "new_time is required."})

    target, err = _authorize_appointment(appointment_id, phone, patient_name, action="reschedule")
    if err:
        return json.dumps(err)

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
        audit("tool.reschedule_appointment", actor="agent",
              subject=subject_ref(patient_id=str(target.get("patient_id") or "")),
              detail={"appointment_id": target.get("appointment_id")})
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
