import json
from typing import Optional

from database import _enrich_appointment, build_availability_suggestions


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


def _slot_refusal(
    doctor: dict,
    preferred_time: str,
    refusal: dict,
    *,
    exclude_id: str = "",
) -> str:
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
