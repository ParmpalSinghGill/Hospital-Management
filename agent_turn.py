"""Shared turn helpers for CLI and voice Cascade.

Voice Cascade should call ``run_turn`` so Deepgram STT/TTS wrap the *same*
LangGraph agent that ``Main.py`` already uses — not a separate LLM bot.
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

# DeepSeek sometimes emits tool calls as fullwidth-pipe DSML markup in plain text.
_FULLWIDTH_PIPE = "\uff5c"  # ｜
_DSML_BLOCK = re.compile(
    rf"<{_FULLWIDTH_PIPE}{_FULLWIDTH_PIPE}DSML{_FULLWIDTH_PIPE}{_FULLWIDTH_PIPE}[^>]*>"
    rf".*?"
    rf"(?:</{_FULLWIDTH_PIPE}{_FULLWIDTH_PIPE}DSML{_FULLWIDTH_PIPE}{_FULLWIDTH_PIPE}[^>]*>|$)",
    re.DOTALL | re.IGNORECASE,
)
_DSML_LOOSE = re.compile(
    rf"<[^>\n]*DSML[^>\n]*>.*?(?:</[^>\n]*DSML[^>\n]*>|$)",
    re.DOTALL | re.IGNORECASE,
)
_ASCII_TOOL_XML = re.compile(
    r"<(?:tool_calls?|function_calls?|invoke|parameter)\b[^>]*>.*?"
    r"(?:</(?:tool_calls?|function_calls?|invoke|parameter)>|$)",
    re.DOTALL | re.IGNORECASE,
)
_ANGLE_TAG = re.compile(r"</?[^>\n]{1,80}>")
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MARKDOWN_BOLD_U = re.compile(r"__(.+?)__", re.DOTALL)
_MARKDOWN_ITALIC = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)", re.DOTALL)
_MARKDOWN_ITALIC_U = re.compile(r"(?<!\w)_(.+?)_(?!\w)", re.DOTALL)
_MARKDOWN_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_HEADING = re.compile(r"(?m)^#{1,6}\s*")
_MULTI_NL = re.compile(r"\n{3,}")

# Filler left after stripping a DSML tool dump — not a real user-facing answer.
_FILLER_ONLY = re.compile(
    r"^(?:ok(?:ay)?|sure|thanks?(?:,?\s+\w+)?|"
    r"let me (?:check|look|get|register|book|find|see|confirm|search|pull)[^.]{0,80}|"
    r"i(?:'ll| will) (?:check|look|get|register|book|find)[^.]{0,80}|"
    r"you(?:'re| are) all registered[^.]{0,40}"
    r")\.?\s*$",
    re.IGNORECASE,
)

_FAKE_TOOLS = (
    "get_availability",
    "get_available_slots",
    "available_slots",
    "register_patient",
    "create_patient",
    "check_slot",
    "find_slots",
)

# Bot invents a human handoff instead of finishing booking — treat as bad reply.
_FAKE_HANDOFF = re.compile(
    r"(?:our|the)\s+team\s+will\s+(?:contact|call|get\s+back)|"
    r"will\s+contact\s+you\s+shortly|"
    r"(?:in\s+touch\s+with\s+you|get\s+back\s+to\s+you)\s+shortly|"
    r"they(?:'ll| will)\s+be\s+in\s+touch|"
    r"forward(?:ed|ing)?\s+your\s+request|"
    r"(?:someone|a\s+(?:colleague|representative|staff))\s+will\s+(?:contact|call|reach)|"
    r"(?:i(?:'ve| have)\s+)?(?:forwarded|passed)\s+(?:your|this)\s+(?:request|details)|"
    r"call\s+you\s+back\s+shortly|"
    r"we(?:'ll| will)\s+(?:have\s+)?(?:someone|our\s+team)\s+(?:call|contact)",
    re.IGNORECASE,
)

# Model falsely claims tools are missing (usually wrong agent tool list / DSML fail).
_FAKE_MISSING_TOOL = re.compile(
    r"(?:don'?t|do not|cannot|can'?t)\s+have\s+(?:a\s+)?tool|"
    r"(?:don'?t|do not|cannot|can'?t)\s+have\s+the\s+ability\s+to\s+(?:cancel|reschedule|book|look\s*up)|"
    r"no\s+tool\s+to\s+(?:cancel|reschedule|book|look\s*up)|"
    r"unable\s+to\s+(?:cancel|reschedule)\s+appointments|"
    r"i\s+(?:don'?t|do not)\s+have\s+(?:a\s+)?tool\s+to\s+look\s*up|"
    r"i\s+(?:don'?t|do not)\s+have\s+(?:the\s+)?(?:ability|permission|access)\s+to\s+"
    r"(?:cancel|reschedule|book|look\s*up|check)",
    re.IGNORECASE,
)


def looks_like_tool_leak(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return (
        "dsml" in t
        or "<tool_call" in t
        or "<invoke" in t
        or _FULLWIDTH_PIPE in text
        or any(name in t for name in _FAKE_TOOLS)
    )


def looks_like_fake_handoff(text: str) -> bool:
    if not text:
        return False
    return bool(_FAKE_HANDOFF.search(text))


def looks_like_fake_missing_tool(text: str) -> bool:
    if not text:
        return False
    return bool(_FAKE_MISSING_TOOL.search(text))


def is_incomplete_tool_reply(text: str) -> bool:
    """True when the spoken reply is only a 'let me check…' preamble after a fake tool dump."""
    if not text:
        return True
    cleaned = sanitize_assistant_reply(text)
    if not cleaned:
        return True
    if looks_like_tool_leak(text):
        return True
    if looks_like_fake_handoff(cleaned):
        return True
    if looks_like_fake_missing_tool(cleaned):
        return True
    if _FILLER_ONLY.match(cleaned.strip()):
        return True
    return False


def sanitize_assistant_reply(text: str) -> str:
    """Strip markdown and leaked tool-call markup before chat/TTS."""
    if not text:
        return ""
    out = str(text)

    # Remove DeepSeek DSML / XML tool dumps first (keep any prose before them).
    out = _DSML_BLOCK.sub("", out)
    out = _DSML_LOOSE.sub("", out)
    out = _ASCII_TOOL_XML.sub("", out)
    # Drop any remaining DSML-looking tags / pipes
    out = re.sub(rf"{_FULLWIDTH_PIPE}+", "", out)
    out = _ANGLE_TAG.sub("", out)

    # Markdown → plain text
    out = _MARKDOWN_BOLD.sub(r"\1", out)
    out = _MARKDOWN_BOLD_U.sub(r"\1", out)
    out = _MARKDOWN_ITALIC.sub(r"\1", out)
    out = _MARKDOWN_ITALIC_U.sub(r"\1", out)
    out = _MARKDOWN_CODE.sub(r"\1", out)
    out = _MARKDOWN_HEADING.sub("", out)
    out = out.replace("**", "").replace("__", "")

    out = _MULTI_NL.sub("\n\n", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()


# Back-compat alias
strip_markdown_for_speech = sanitize_assistant_reply


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts).strip()
    if isinstance(msg, tuple) and len(msg) >= 2:
        return str(msg[1] or "")
    return ""


def _is_ai_message(msg: Any) -> bool:
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    if isinstance(msg, tuple) and msg:
        role = msg[0]
    role_s = str(role or "").lower()
    return role_s in ("ai", "assistant")


def _has_tool_calls(msg: Any) -> bool:
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return True
    additional = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(additional, dict) and additional.get("tool_calls"):
        return True
    return False


def extract_assistant_text(result: dict[str, Any]) -> str:
    """Pick the last usable assistant utterance from the latest turn only."""
    messages = result.get("messages", []) or []

    # Only inspect messages after the last user turn (avoids returning stale replies).
    start = 0
    for i, msg in enumerate(messages):
        role = str(getattr(msg, "type", getattr(msg, "role", "")) or "").lower()
        if isinstance(msg, tuple) and msg:
            role = str(msg[0] or "").lower()
        if role in ("human", "user"):
            start = i + 1
    turn_msgs = list(messages[start:]) if start < len(messages) else list(messages)

    for msg in reversed(turn_msgs):
        if not _is_ai_message(msg):
            continue
        # Intermediate ReAct step with structured tool_calls — keep looking.
        if _has_tool_calls(msg):
            continue

        raw = _message_text(msg)
        if not raw or not raw.strip():
            continue
        if looks_like_tool_leak(raw) or is_incomplete_tool_reply(raw):
            continue

        cleaned = sanitize_assistant_reply(raw)
        if cleaned and not is_incomplete_tool_reply(cleaned) and not looks_like_fake_handoff(cleaned):
            return cleaned

    return ""


def result_needs_tool_recovery(result: dict[str, Any]) -> bool:
    """True if the latest turn ended on a fake DSML dump or unfinished tool call."""
    messages = result.get("messages", []) or []
    if not messages:
        return True

    start = 0
    for i, msg in enumerate(messages):
        role = str(getattr(msg, "type", getattr(msg, "role", "")) or "").lower()
        if isinstance(msg, tuple) and msg:
            role = str(msg[0] or "").lower()
        if role in ("human", "user"):
            start = i + 1
    turn_msgs = list(messages[start:]) if start < len(messages) else list(messages)
    if not turn_msgs:
        return True

    last = turn_msgs[-1]
    last_role = str(getattr(last, "type", getattr(last, "role", "")) or "").lower()

    # Ended right after a tool result with no final spoken AI reply.
    if last_role in ("tool", "function"):
        return True

    if _is_ai_message(last):
        if _has_tool_calls(last):
            return True
        raw = _message_text(last)
        if looks_like_tool_leak(raw) or is_incomplete_tool_reply(raw) or looks_like_fake_handoff(raw):
            return True
        cleaned = sanitize_assistant_reply(raw)
        if not cleaned or is_incomplete_tool_reply(cleaned) or looks_like_fake_handoff(cleaned):
            return True
        return False

    text = extract_assistant_text(result)
    return not text or is_incomplete_tool_reply(text) or looks_like_fake_handoff(text)


_RECOVERY_NUDGE = (
    "Stop. Do not print tool XML, DSML, or fake tools. "
    "There is NO register_patient, get_available_slots, or get_availability tool. "
    "Do NOT say the team will contact the patient or that you forwarded anything. "
    "Do NOT say you lack cancel/reschedule/book tools — you have them. "
    "You must finish the request yourself. "
    "Allowed tools only (via the normal tool interface): "
    "lookup_patient, save_patient, list_doctors, book_appointment, cancel_appointment, "
    "reschedule_appointment, get_prescriptions. "
    "SOFT HUMAN FLOW: greet first if needed; ask what brings them in — never a menu "
    "(book/cancel/reschedule). When intent is clear, ask phone, confirm, then name, "
    "then help from intent + patient history. "
    "If they already have an appointment, tell them and offer to move/cancel — "
    "do not push a new booking. If no visit for their need, offer to set one up softly. "
    "Medicine / timing confusion → get_prescriptions after phone+name. "
    "If the patient already has an appointment and wants a new time, call reschedule_appointment. "
    "If they asked to cancel, call cancel_appointment (id or phone). "
    "After phone and name are both confirmed for a new patient, call save_patient. "
    "Always confirm the full name aloud and get a yes before save_patient or booking. "
    "If you need a tool, call it properly and then speak the result to the user. "
    "Otherwise speak one short plain sentence. "
    "Prefer bare questions like 'What brings you in today?' or 'What's your phone number?' "
    "— never long 'let me look you up' narration. "
    "Never invent availability lists or human handoffs. "
    "If book_appointment or reschedule_appointment returns time_conflict, "
    "offer nearest_times then alternate_doctors from that tool result only. "
    "If list_doctors returns preferred_time_unavailable, offer nearest_times "
    "in the same department — do not switch departments for lunch/busy slots. "
    "If day_only_preference=true (patient said 'today' without a clock), doctors ARE "
    "available — offer nearest_times and ask which time; never say the day is unavailable. "
    "After department is known, call lookup_patient(phone=..., department=...) and follow "
    "booking_guidance: inform_existing_soon, offer_prepone, or offer_new_booking. "
    "If the user just said yes/yeah/ok/sure/go ahead to your last confirmation question, "
    "do NOT repeat that question — call the next tool or ask the next field only."
)

_HANDOFF_RECOVERY_NUDGE = (
    "Do not say anyone will contact them, and do not say you lack tools. "
    "You already have lookup_patient (returns active_appointment with time), "
    "cancel_appointment, reschedule_appointment, and book_appointment. "
    "If they ask what time their appointment is, call lookup_patient(phone=...) "
    "and read active_appointment.time and appointment_id aloud — never invent a missing tool. "
    "If they want a new time, call reschedule_appointment. "
    "Ask only ONE short question if you still need the new time."
)


class TurnResult(NamedTuple):
    text: str
    agent: str = ""


def _tool_message_payload(msg: Any) -> tuple[str, Any]:
    """Return (tool_name, parsed_json_or_raw) for a ToolMessage-like object."""
    name = str(getattr(msg, "name", None) or getattr(msg, "tool", None) or "").strip()
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        name = str(msg.get("name") or msg.get("tool") or name).strip()
        content = msg.get("content")
    if isinstance(content, (dict, list)):
        return name, content
    text = content if isinstance(content, str) else str(content or "")
    try:
        return name, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return name, text


def _link_patient_from_graph_result(call_id: str, result: Any) -> None:
    """Tag the call log as soon as the phone resolves to a patient (number confirmed)."""
    if not call_id or not isinstance(result, dict):
        return
    try:
        from conversation_log import link_call_patient
    except Exception:
        return
    messages = result.get("messages") or []
    for msg in reversed(list(messages)):
        name, payload = _tool_message_payload(msg)
        if name not in (
            "lookup_patient",
            "save_patient",
            "book_appointment",
            "cancel_appointment",
            "reschedule_appointment",
        ):
            continue
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue

        patient = payload.get("patient") if isinstance(payload.get("patient"), dict) else None
        appt = payload.get("appointment") if isinstance(payload.get("appointment"), dict) else None
        pid = ""
        phone = ""
        aid = ""
        if patient:
            pid = str(patient.get("patient_id") or "")
            phone = str(patient.get("phone") or "")
        if appt:
            pid = pid or str(appt.get("patient_id") or "")
            aid = str(appt.get("appointment_id") or "")
            phone = phone or str(appt.get("patient_phone") or "")

        # Phone lookup success = number confirmed → always link (even if name still needed).
        if name == "lookup_patient" and not pid and not phone:
            continue
        if pid or phone:
            link_call_patient(
                call_id,
                patient_id=pid,
                phone=phone,
                appointment_id=aid,
            )
            return


def _messages_after_last_user(messages: list[Any] | None) -> list[Any]:
    """Return only messages after the most recent user turn."""
    if not messages:
        return []
    start = 0
    for i, msg in enumerate(messages):
        role = str(getattr(msg, "type", None) or getattr(msg, "role", None) or "").lower()
        if isinstance(msg, tuple) and msg:
            role = str(msg[0] or "").lower()
        if role in ("human", "user"):
            start = i + 1
    return list(messages[start:]) if start < len(messages) else list(messages)


def _log_turn_tool_calls(
    result: Any,
    *,
    call_id: str,
    agent: str = "",
    source: str = "langgraph",
) -> None:
    try:
        from conversation_log import record_tool_calls_from_messages

        msgs = _messages_after_last_user((result or {}).get("messages") or [])
        record_tool_calls_from_messages(
            msgs,
            source=source,
            call_id=call_id,
            agent=agent or str((result or {}).get("last_agent") or ""),
        )
    except Exception:
        pass


def run_turn(
    app: Any,
    user_input: str,
    thread_id: str,
    *,
    recursion_limit: int = 25,
    call_id: str | None = None,
) -> TurnResult:
    """Run one user turn on the compiled hospital LangGraph app."""
    log_id = call_id or thread_id
    try:
        from conversation_log import set_current_call_id

        set_current_call_id(log_id)
    except Exception:
        pass

    config = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    result = app.invoke({"messages": [("user", user_input)]}, config=config)
    text = extract_assistant_text(result)
    agent = str((result or {}).get("last_agent") or "")
    # Some LangGraph/checkpointer combinations omit non-message state keys from
    # invoke() even though the value was persisted. Read the checkpoint as a
    # fallback so every logged assistant reply retains its actual node name.
    if not agent:
        try:
            snapshot = app.get_state(config)
            values = getattr(snapshot, "values", None) or {}
            agent = str(values.get("last_agent") or "")
        except Exception:
            pass
    _log_turn_tool_calls(result, call_id=log_id, agent=agent)

    # DeepSeek often finishes with DSML text instead of executing tools — nudge and retry.
    attempts = 0
    while attempts < 2 and (
        not text
        or result_needs_tool_recovery(result)
        or is_incomplete_tool_reply(text)
        or looks_like_fake_handoff(text)
        or looks_like_fake_missing_tool(text or "")
    ):
        attempts += 1
        nudge = (
            _HANDOFF_RECOVERY_NUDGE
            if looks_like_fake_handoff(text or "") or looks_like_fake_missing_tool(text or "")
            else _RECOVERY_NUDGE
        )
        result = app.invoke(
            {"messages": [("user", nudge)]},
            config=config,
        )
        text = extract_assistant_text(result)
        agent = str((result or {}).get("last_agent") or agent)
        if not agent:
            try:
                snapshot = app.get_state(config)
                values = getattr(snapshot, "values", None) or {}
                agent = str(values.get("last_agent") or "")
            except Exception:
                pass
        _log_turn_tool_calls(result, call_id=log_id, agent=agent)
        if (
            text
            and not is_incomplete_tool_reply(text)
            and not looks_like_tool_leak(text)
            and not looks_like_fake_handoff(text)
            and not looks_like_fake_missing_tool(text)
        ):
            break

    if (
        not text
        or is_incomplete_tool_reply(text)
        or looks_like_tool_leak(text)
        or looks_like_fake_handoff(text)
        or looks_like_fake_missing_tool(text or "")
    ):
        text = (
            "Sorry, I had trouble with that step. "
            "Just to confirm — should I reschedule your existing appointment to the new time, "
            "or cancel it and book a new one?"
        )
        agent = agent or "recovery"
    else:
        _link_patient_from_graph_result(log_id, result)

    try:
        from conversation_log import note_turn_meta

        note_turn_meta(log_id, agent_name=agent)
    except Exception:
        pass

    return TurnResult(text=text, agent=agent)
