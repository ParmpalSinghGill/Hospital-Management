import json
import logging
from typing import Optional

from langchain_core.tools import tool

from database import (
    _load_doctors,
    department_matches,
    find_nearest_available_times,
    is_within_clinic_hours,
    normalize_appointment_time,
    rank_doctors_for_preferred_time,
    resolve_appointment_day,
)


@tool
def list_doctors(
    department: Optional[str] = None,
    query: Optional[str] = None,
    exclude_doctor: Optional[str] = None,
    preferred_time: Optional[str] = None,
    limit: int = 10,
) -> str:
    """List doctors from the directory, optionally filtered by department or a name query."""
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
