import json
import logging
from typing import Optional

from langchain_core.tools import tool

from database import (
    _complete_past_appointments_for_patient,
    _enrich_appointment,
    _find_active_appointment_for_patient,
    _find_patient_by_phone,
    _get_or_create_patient,
    _names_match,
    _normalize_phone,
    _patient_past_doctors,
    _phone_is_complete,
    _resolve_patient,
    build_department_booking_guidance,
    classify_appointment_timing,
)


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
    """Create or update a patient record after phone and name are confirmed."""
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
