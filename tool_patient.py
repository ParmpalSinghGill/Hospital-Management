import json
import logging
from typing import Optional

from langchain_core.tools import tool

from audit_log import audit, subject_ref
from phi_redact import mask_phone
from rate_limit import RateLimiter

# Per-session identity attempts: 8 lookups / 10 min, 3 failed name checks / 10 min.
_LOOKUP_LIMIT = RateLimiter(max_events=8, window_secs=600)
_MISMATCH_LIMIT = RateLimiter(max_events=3, window_secs=600)


def _session_key() -> str:
    try:
        from conversation_log import get_current_call_id

        return get_current_call_id() or "anon"
    except Exception:  # noqa: BLE001
        return "anon"

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

    Identity = phone AND full name, both supplied by the caller. The tool never
    reveals the name or phone on file. Flow: ask phone → lookup_patient(phone=...)
    → if is_returning, ask their full name → lookup_patient(phone=..., patient_name=...)
    → verified=true returns the record. If is_new, ask full name then save_patient.
    """
    logging.info(
        "lookup_patient id=%s phone=%s name=%s department=%s",
        patient_id,
        mask_phone(phone),
        "yes" if (patient_name or "").strip() else "no",
        department,
    )
    skey = _session_key()
    if not _LOOKUP_LIMIT.allow(skey):
        audit("tool.lookup_patient", actor="agent", session_id=skey, outcome="rate_limited")
        return json.dumps({
            "ok": False,
            "message": "Too many lookup attempts in this conversation. Please try again later.",
            "rate_limited": True,
        })

    phone_s = (phone or "").strip()
    name_s = (patient_name or "").strip()
    pid_s = (patient_id or "").strip()

    if not any([pid_s, phone_s, name_s]):
        return json.dumps({
            "ok": False,
            "message": "Ask for the phone number first. Names can be the same for different people.",
            "needs_phone": True,
        })

    if pid_s and not phone_s:
        # A patient id alone is guessable; require the phone as a second factor.
        return json.dumps({
            "ok": False,
            "message": "Ask for the phone number registered with this patient id to verify.",
            "needs_phone": True,
        })

    resolved = _resolve_patient(
        patient_id=pid_s or None,
        phone=phone_s or None,
        patient_name=name_s or None,
        require_name_with_phone=False,
    )

    if resolved.get("name_mismatch"):
        audit(
            "tool.lookup_patient", actor="agent", session_id=skey, outcome="name_mismatch",
            subject=subject_ref(phone=phone_s),
        )
        if not _MISMATCH_LIMIT.allow(skey):
            return json.dumps({
                "ok": False,
                "message": (
                    "Identity could not be verified after several attempts. Ask the caller to "
                    "contact the front desk directly."
                ),
                "name_mismatch": True,
                "locked": True,
                "needs_name": False,
                "needs_phone": False,
            })
        return json.dumps({
            "ok": False,
            "message": resolved.get("message"),
            "name_mismatch": True,
            "is_returning": True,
            "needs_phone": not bool(phone_s),
            "needs_name": True,
        })

    if resolved.get("is_returning") and resolved.get("needs_name") and not resolved.get("patient"):
        return json.dumps({
            "ok": True,
            "is_returning": True,
            "is_new": False,
            "verified": False,
            "needs_name": True,
            "needs_phone": False,
            "message": resolved.get("message"),
        })

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
        }
        if resolved.get("incomplete_phone"):
            payload["incomplete_phone"] = True
        if resolved.get("name_match_count") is not None:
            payload["name_match_count"] = resolved.get("name_match_count")
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

    # Reaching here means phone + name matched (or id + phone matched): verified.
    name_confirmed = True
    confirm_name = patient.get("name")
    if active_enriched:
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

    audit(
        "tool.lookup_patient", actor="agent", session_id=skey,
        subject=subject_ref(patient_id=str(patient.get("patient_id") or "")),
    )
    payload = {
        "ok": True,
        "is_returning": True,
        "is_new": False,
        "verified": True,
        "needs_name": False,
        "needs_phone": False,
        "message": msg,
        "patient": patient,
        "past_doctors": past,
        "last_doctor": (past[0] if past else None),
        "active_appointment": active_enriched,
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
    logging.info("save_patient phone=%s", mask_phone(phone))
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
        audit("tool.save_patient", actor="agent", session_id=_session_key(), outcome="phone_conflict",
              subject=subject_ref(phone=ph))
        return json.dumps({
            "ok": False,
            "message": (
                "This phone number is already registered under a different name. Ask the caller "
                "to re-check the digits; if the number is right, the front desk must update "
                "the record — do not create or reveal anything."
            ),
            "phone_conflict": True,
            "needs_phone": True,
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
    audit("tool.save_patient", actor="agent", session_id=_session_key(),
          outcome="created" if created else "updated",
          subject=subject_ref(patient_id=str((patient or {}).get("patient_id") or "")))
    return json.dumps({
        "ok": True,
        "message": "Patient saved" if created else "Patient updated",
        "created": created,
        "patient": patient,
    })
