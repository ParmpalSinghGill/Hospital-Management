"""Intent routing helpers for the hospital LangGraph."""
from __future__ import annotations

import re
from typing import Any, List, Literal


def _message_plain(msg: Any) -> str:
    if isinstance(msg, tuple) and len(msg) >= 2:
        return str(msg[1] or "")
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return str(content or "")


def _message_role(msg: Any) -> str:
    if isinstance(msg, tuple) and msg:
        return str(msg[0] or "").lower()
    return str(getattr(msg, "type", getattr(msg, "role", "")) or "").lower()


def _is_short_affirmation(text: str) -> bool:
    t = (text or "").strip().lower().replace("!", "").replace("?", "")
    t = " ".join(t.replace(",", " ").split())
    if not t:
        return False
    # Normalize common multi-word confirms before token checks.
    for phrase in (
        "please go ahead",
        "go ahead please",
        "go ahead",
        "sounds good",
        "that's right",
        "thats right",
        "that is correct",
        "all right",
    ):
        t = t.replace(phrase, " yes ")
    t = " ".join(t.split())
    atoms = {
        "yes", "yeah", "yep", "yup", "ya", "yea", "ok", "okay", "sure",
        "correct", "right", "alright", "please",
    }
    if t in atoms:
        return True
    # "yeah. yes." / "yes yes" / "sure please"
    parts = [p for p in re.split(r"[.\s]+", t) if p]
    return bool(parts) and all(p in atoms for p in parts)


def _is_greeting_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.fullmatch(
            r"(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)"
            r"([!?.]|\s+(there|again))*[!?.\s]*",
            t,
        )
    )


_DEPT_MEDICINE_RE = re.compile(
    r"\b(?:family|internal|general|emergency|nuclear|preventive|sports)\s+medicine\b",
    re.IGNORECASE,
)


_PRESCRIPTION_INTENT_RE = re.compile(
    r"\b(?:prescriptions?|medications?|medicines?|dosage|how do i take|when do i take|what did the doctor)\b",
    re.IGNORECASE,
)


def _wants_prescriptions(text: str) -> bool:
    """True for medicine/Rx intent — not department names like 'family medicine'."""
    cleaned = _DEPT_MEDICINE_RE.sub(" ", text or "")
    return bool(_PRESCRIPTION_INTENT_RE.search(cleaned))


def _sticky_route_from_history(
    messages: List[Any],
    *,
    last_agent: str = "",
) -> Literal["general", "booking", "cancelling", "rescheduling", "prescriptions"] | None:
    """Keep mid-flow conversations on the specialist agent (esp. booking)."""
    recent = messages[-16:] if messages else []
    blob = " ".join(_message_plain(m).lower() for m in recent)
    last_user = ""
    for m in reversed(recent):
        if _message_role(m) in ("human", "user"):
            last_user = _message_plain(m).lower()
            break

    # Bare greetings ALWAYS reopen on general — even if an old thread still has
    # last_agent=booking / phone/pain markers in checkpoint history.
    if _is_greeting_only(last_user):
        return "general"

    if any(k in last_user for k in ("cancel appointment", "cancel my", "i want to cancel")):
        return "cancelling"
    if any(k in last_user for k in ("reschedule", "move my appointment", "change the time")):
        return "rescheduling"
    if _wants_prescriptions(last_user):
        return "prescriptions"

    prev = (last_agent or "").strip().lower()
    # Stay on the same specialist for yes/ok and other short mid-flow answers.
    # Do NOT treat greetings as mid-flow (handled above).
    if prev in ("booking", "cancelling", "rescheduling", "prescriptions"):
        if _is_short_affirmation(last_user):
            return prev  # type: ignore[return-value]
        words = last_user.split()
        if len(words) <= 6 and not any(
            k in last_user for k in ("cancel", "reschedule", "prescription")
        ) and not _wants_prescriptions(last_user):
            return prev  # type: ignore[return-value]

    booking_markers = (
        "book",
        "appointment",
        "phone",
        "number is",
        "full name",
        "just to confirm",
        "doctor",
        "department",
        "headache",
        "general medicine",
        "family medicine",
        "patient id",
        "tomorrow",
        " am",
        " pm",
        "pain",
        "stomach",
        "belly",
        "tooth",
        "dental",
        "fever",
        "sick",
        "hurt",
        "ache",
        "nausea",
        "vomit",
        "cough",
        "see a doctor",
        "need a doctor",
        "gastro",
    )
    # After soft intent is known (symptoms / visit talk), stay on booking —
    # unless still on general and they only answered a vague chit-chat line.
    if any(k in blob for k in booking_markers):
        if any(k in last_user for k in ("cancel", "reschedule")) or _wants_prescriptions(last_user):
            return None
        # Keep general for one soft "what brings you" answer only if no medical need yet.
        if prev == "general" and not any(k in last_user for k in _SYMPTOM_BOOKING_HINTS) and not any(
            k in last_user for k in ("doctor", "appointment", "visit", "see someone", "check up", "checkup")
        ):
            return "general"
        return "booking"
    return None


_SYMPTOM_BOOKING_HINTS = (
    "pain",
    "ache",
    "hurt",
    "fever",
    "sick",
    "stomach",
    "belly",
    "tooth",
    "dental",
    "headache",
    "nausea",
    "vomit",
    "cough",
    "see a doctor",
    "need a doctor",
    "unwell",
    "illness",
)

