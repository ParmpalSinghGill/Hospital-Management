"""Query, delete, and seed helpers for chat sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chat_log_core as core
from chat_log_turns import _empty_interaction, _interactions

# Path/helpers resolved via core.* so tests can redirect chat dirs at runtime.
CHATS_DIR = core.CHATS_DIR  # noqa: F841 — kept for type checkers; prefer core.CHATS_DIR


def list_recent_calls(limit: int = 20) -> list[dict[str, Any]]:
    core._ensure_layout()
    files = sorted(
        list(core.SESSIONS_DIR.glob("sess-*.json")) + list(core.SESSIONS_DIR.glob("call-*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        data = core._read(path)
        if not data:
            continue
        interactions = _interactions(data)
        out.append(
            {
                "file": path.name,
                "session_id": data.get("session_id") or data.get("call_id"),
                "user_id": data.get("user_id"),
                "patient_id": data.get("patient_id") or (data.get("meta") or {}).get("patient_id"),
                "session_start_time": data.get("session_start_time") or data.get("started_at"),
                "session_end_time": data.get("session_end_time") or data.get("ended_at"),
                "interactions": len(interactions),
            }
        )
    return out

def _iter_chat_files() -> list[Path]:
    core._ensure_layout()
    return sorted(
        list(core.SESSIONS_DIR.glob("sess-*.json")) + list(core.SESSIONS_DIR.glob("call-*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 1)


def _timing_samples_from_interactions(
    interactions: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    text_samples: list[float] = []
    audio_samples: list[float] = []
    for ix in interactions:
        if not isinstance(ix, dict):
            continue
        timings = ix.get("client_timings")
        if not isinstance(timings, dict):
            continue
        raw_text = timings.get("first_text_latency_ms")
        raw_audio = timings.get("first_audio_latency_ms")
        try:
            if raw_text is not None:
                text_samples.append(float(raw_text))
        except (TypeError, ValueError):
            pass
        try:
            if raw_audio is not None:
                audio_samples.append(float(raw_audio))
        except (TypeError, ValueError):
            pass
    return text_samples, audio_samples


def _bucket(samples: list[float], sessions_hit: int = 0) -> dict[str, Any]:
    return {
        "avg_ms": _avg(samples),
        "count": len(samples),
        "min_ms": round(min(samples), 1) if samples else None,
        "max_ms": round(max(samples), 1) if samples else None,
        "sessions": sessions_hit,
    }


def session_response_timings(session_id: str) -> dict[str, Any] | None:
    """Average first-text / first-speech for one session under ``chats/sessions/``."""
    sid = (session_id or "").strip()
    if not sid:
        return None
    for path in _iter_chat_files():
        data = core._read(path)
        if not data:
            continue
        file_sid = str(data.get("session_id") or data.get("call_id") or "").strip()
        if file_sid != sid and not path.stem.endswith(sid):
            continue
        interactions = _interactions(data)
        text_samples, audio_samples = _timing_samples_from_interactions(interactions)
        return {
            "session_id": file_sid or sid,
            "user_id": data.get("user_id"),
            "turn_count": len(interactions),
            "first_text": _bucket(text_samples),
            "first_speech": _bucket(audio_samples),
            "avg_first_text_ms": _avg(text_samples),
            "avg_first_speech_ms": _avg(audio_samples),
        }
    return None


def summarize_response_timings(*, include_sessions: bool = False) -> dict[str, Any]:
    """Average first-text / first-audio latency across all ``chats/sessions/`` files.

    Reads ``interactions[].client_timings`` written by the browser for each turn.
    """
    core._ensure_layout()
    text_samples: list[float] = []
    audio_samples: list[float] = []
    user_ids: set[str] = set()
    sessions_with_text = 0
    sessions_with_audio = 0
    session_count = 0
    turn_count = 0
    per_session: list[dict[str, Any]] = []

    for path in _iter_chat_files():
        data = core._read(path)
        if not data:
            continue
        session_count += 1
        uid = str(data.get("user_id") or "").strip()
        if uid:
            user_ids.add(uid)
        interactions = _interactions(data)
        turn_count += len(interactions)
        sid = str(data.get("session_id") or data.get("call_id") or path.stem)
        s_text, s_audio = _timing_samples_from_interactions(interactions)
        text_samples.extend(s_text)
        audio_samples.extend(s_audio)
        if s_text:
            sessions_with_text += 1
        if s_audio:
            sessions_with_audio += 1
        if include_sessions:
            mtime = 0.0
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            per_session.append(
                {
                    "session_id": sid,
                    "user_id": uid or None,
                    "file": path.name,
                    "mtime": mtime,
                    "turn_count": len(interactions),
                    "first_text": _bucket(s_text),
                    "first_speech": _bucket(s_audio),
                    "avg_first_text_ms": _avg(s_text),
                    "avg_first_speech_ms": _avg(s_audio),
                }
            )

    if include_sessions:
        per_session.sort(key=lambda r: float(r.get("mtime") or 0), reverse=True)

    out = {
        "ok": True,
        "source": "chats/sessions",
        "session_count": session_count,
        "turn_count": turn_count,
        "user_count": len(user_ids),
        "first_text": _bucket(text_samples, sessions_with_text),
        "first_speech": _bucket(audio_samples, sessions_with_audio),
    }
    if include_sessions:
        out["sessions"] = per_session
    return out

def _infer_legacy_agent(user_text: str, bot_text: str, previous: str) -> str:
    """Best-effort specialist label for logs created before agent capture."""
    text = f"{user_text} {bot_text}".lower()
    if any(word in text for word in ("prescription", "medication", "medicines")):
        return "prescriptions"
    if any(word in text for word in ("reschedule", "move my appointment", "change the time")):
        return "rescheduling"
    if "cancel" in text:
        return "cancelling"
    if any(word in text for word in ("book", "appointment", "doctor", "department")):
        return previous or "booking"
    if previous:
        return previous
    return "general"

def _chat_preview_turns(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview = []
    previous_agent = ""
    for ix in interactions:
        payload = ix.get("payload") if isinstance(ix.get("payload"), dict) else {}
        user_t = payload.get("user_input_text") or ""
        bot_t = payload.get("bot_response_text") or ""
        if user_t or bot_t:
            recorded_agent = str(ix.get("agent") or "").strip()
            inferred_agent = ""
            # Only label an assistant turn. Empty STT fragments have no agent reply.
            if not recorded_agent and bot_t:
                inferred_agent = _infer_legacy_agent(user_t, bot_t, previous_agent)
            agent = recorded_agent or inferred_agent
            if agent:
                previous_agent = agent
            preview.append(
                {
                    "turn_number": ix.get("turn_number"),
                    "user": user_t,
                    "assistant": bot_t,
                    "agent": agent,
                    "agent_inferred": bool(inferred_agent),
                }
            )
    return preview

def _normalize_digits(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())

def _interaction_blob(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for ix in _interactions(data):
        payload = ix.get("payload") if isinstance(ix.get("payload"), dict) else {}
        parts.append(str(payload.get("user_input_text") or ""))
        parts.append(str(payload.get("bot_response_text") or ""))
    return " ".join(parts)


def _is_seed_chat(data: dict[str, Any], *, path_name: str = "", session_id: str = "") -> bool:
    pipeline = str(data.get("pipeline_mode") or "").strip().lower()
    channel = str(data.get("channel") or "").strip().lower()
    sid = (session_id or str(data.get("session_id") or data.get("call_id") or "")).lower()
    name = path_name.lower()
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    kind = str(meta.get("kind") or "").strip().lower()
    return (
        pipeline == "seed"
        or channel in {"admin_seed", "seed"}
        or sid.startswith("book-")
        or "book-apt-" in name
        or kind == "appointment_booking" and pipeline in {"", "seed", "booking"} and sid.startswith("book-")
    )


def _chat_matches_patient(
    data: dict[str, Any],
    *,
    patient_id: str,
    phone_digits: str,
    appointment_ids: set[str],
    patient_name: str = "",
) -> bool:
    """Match tagged calls, or live calls where this patient was clearly discussed."""
    from database import _normalize_phone
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    chat_pid = str(data.get("patient_id") or meta.get("patient_id") or "").strip().upper()
    if patient_id and chat_pid == patient_id.upper():
        return True

    aid = str(data.get("appointment_id") or meta.get("appointment_id") or "").strip().upper()
    if aid and aid in appointment_ids:
        return True

    blob = _interaction_blob(data)
    blob_digits = _normalize_digits(blob)
    low = blob.lower()

    # Appointment id spoken/typed in the live transcript.
    for appt_id in appointment_ids:
        if appt_id and appt_id.lower() in low:
            return True

    chat_phone = _normalize_digits(str(data.get("phone") or meta.get("phone") or ""))
    if phone_digits and len(phone_digits) >= 8:
        if chat_phone and (
            chat_phone == phone_digits
            or chat_phone.endswith(phone_digits)
            or phone_digits.endswith(chat_phone)
        ):
            return True
        # Fallback for older/unlinked calls: phone spoken + bot acknowledged the lookup.
        if phone_digits in blob_digits:
            if any(
                marker in low
                for marker in (
                    "found you",
                    "look you up",
                    "patient id",
                    "is that correct",
                    "phone number is",
                    "on file",
                    "रजिस्टर",
                    "रजिस्ट्रेशन",
                    "अपॉइंटमेंट",
                )
            ):
                return True

    # Name-based fallback when STT mangled the phone digits but the bot used the
    # patient's real name while handling a visit.
    name = (patient_name or "").strip()
    if name and len(name) >= 3 and name in blob:
        if any(
            marker in low
            for marker in (
                "appointment",
                "doctor",
                "patient",
                "book",
                "reschedule",
                "cancel",
                "found you",
                "on file",
            )
        ) or any(
            marker in blob
            for marker in (
                "अपॉइंटमेंट",
                "डॉक्टर",
                "रजिस्टर",
                "रजिस्ट्रेशन",
                "फोन",
                "मरीज",
            )
        ):
            return True
    return False


def list_calls_for_patient(
    patient_id: str,
    *,
    include_seed: bool = False,
) -> list[dict[str, Any]]:
    """Return chat sessions for this patient (tagged or clearly discussed), newest first.

    Seed/synthetic booking templates from MakeDataBase are hidden by default so the
    admin page shows real live chats instead of canned scripts.
    """
    from database import _get_patient_by_id, _load_all_appointments, _normalize_phone
    pid = (patient_id or "").strip()
    if not pid:
        return []

    phone_digits = ""
    patient_name = ""
    appointment_ids: set[str] = set()
    try:
        patient = _get_patient_by_id(pid)
        if patient:
            phone_digits = _normalize_phone(str(patient.get("phone") or ""))
            patient_name = str(patient.get("name") or "").strip()
        for a in _load_all_appointments():
            if str(a.get("patient_id") or "").strip().upper() == pid.upper():
                aid = str(a.get("appointment_id") or "").strip().upper()
                if aid:
                    appointment_ids.add(aid)
    except Exception:
        pass

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _iter_chat_files():
        data = core._read(path)
        if not data:
            continue
        sid = str(data.get("session_id") or data.get("call_id") or path.name)
        is_seed = _is_seed_chat(data, path_name=path.name, session_id=sid)
        if is_seed and not include_seed:
            continue
        if not _chat_matches_patient(
            data,
            patient_id=pid,
            phone_digits=phone_digits,
            appointment_ids=appointment_ids,
            patient_name=patient_name,
        ):
            continue
        if sid in seen:
            continue
        seen.add(sid)
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        interactions = _interactions(data)
        kind = str(meta.get("kind") or "").strip()
        pipeline = str(data.get("pipeline_mode") or "").strip()
        if not kind:
            if is_seed:
                kind = "seed:booking"
            elif pipeline in ("cascade", "realtime", "cli"):
                kind = f"live:{pipeline}"
            else:
                kind = pipeline or "live"
        # Surface intent from transcript for admin scanning.
        blob = _interaction_blob(data).lower()
        topic = ""
        if "reschedule" in blob:
            topic = "reschedule"
        elif "cancel" in blob:
            topic = "cancel"
        elif "book" in blob or "appointment" in blob or "अपॉइंटमेंट" in blob:
            topic = "booking"
        out.append(
            {
                "file": path.name,
                "session_id": data.get("session_id") or data.get("call_id"),
                "patient_id": data.get("patient_id") or meta.get("patient_id") or pid,
                "appointment_id": data.get("appointment_id") or meta.get("appointment_id"),
                "kind": kind,
                "topic": topic,
                "pipeline_mode": pipeline,
                "is_seed": is_seed,
                "session_start_time": data.get("session_start_time") or data.get("started_at"),
                "session_end_time": data.get("session_end_time") or data.get("ended_at"),
                "interaction_count": len(interactions),
                "turns": _chat_preview_turns(interactions),
                "timeline": [
                    ev
                    for ev in (data.get("timeline") or [])
                    if isinstance(ev, dict)
                    and str(ev.get("type") or "")
                    in (
                        "interrupt",
                        "interrupt_during_inference",
                        "interrupt_during_tts_push",
                        "barge_in_while_bot_speaking",
                        "queued_while_busy",
                        "transcript_during_bot",
                        "user_turn_started",
                        "user_turn_stopped",
                    )
                ][:40],
            }
        )

    out.sort(key=lambda c: str(c.get("session_start_time") or ""), reverse=True)
    return out

def delete_call(session_id: str) -> bool:
    """Delete a chat log file by session_id (or matching file name). Returns True if removed."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    safe = core._safe_id(sid)
    with core._LOCK:
        removed = False
        # Close open handle if this session is active in-process.
        for key in list(core._OPEN.keys()):
            if key == sid or core._safe_id(key) == safe:
                core._OPEN.pop(key, None)
        for path in list(_iter_chat_files()):
            data = core._read(path)
            file_sid = str((data or {}).get("session_id") or (data or {}).get("call_id") or "")
            if (
                file_sid == sid
                or path.name == sid
                or path.name == f"{sid}.json"
                or path.stem == sid
                or path.stem == safe
                or path.stem.endswith(sid)
                or path.stem.endswith(safe)
            ):
                try:
                    path.unlink(missing_ok=True)
                    removed = True
                except OSError:
                    pass
        return removed

def get_call_detail(session_id: str) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    for path in _iter_chat_files():
        data = core._read(path)
        if not data:
            continue
        if str(data.get("session_id") or data.get("call_id") or "") == sid or path.stem.endswith(sid):
            interactions = _interactions(data)
            turns = []
            for ix in interactions:
                payload = ix.get("payload") if isinstance(ix.get("payload"), dict) else {}
                turns.append(
                    {
                        "turn_number": ix.get("turn_number"),
                        "mode": ix.get("mode"),
                        "agent": ix.get("agent") or "",
                        "user": payload.get("user_input_text") or "",
                        "assistant": payload.get("bot_response_text") or "",
                        "timestamps": ix.get("timestamps") or {},
                    }
                )
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            return {
                "file": path.name,
                "session_id": data.get("session_id") or data.get("call_id"),
                "patient_id": data.get("patient_id") or meta.get("patient_id"),
                "appointment_id": data.get("appointment_id") or meta.get("appointment_id"),
                "session_start_time": data.get("session_start_time"),
                "session_end_time": data.get("session_end_time"),
                "pipeline_mode": data.get("pipeline_mode") or "",
                "turns": turns,
                "timeline": list(data.get("timeline") or []),
                "raw": data,
            }
    return None

def record_booking_chat(
    *,
    patient: dict[str, Any],
    appointment: dict[str, Any],
    transcript: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Write a booking conversation log linked to the patient (for admin history)."""
    from datetime import datetime, timezone

    from chat_log_turns import _empty_interaction, end_call, start_call
    pid = str(patient.get("patient_id") or "")
    aid = str(appointment.get("appointment_id") or "")
    name = str(patient.get("name") or "Patient")
    doctor = str(appointment.get("doctor") or "the doctor")
    when = str(appointment.get("time") or "")
    phone = str(patient.get("phone") or "")

    if transcript is None:
        transcript = [
            ("I need to book an appointment.", "Sure — may I have your phone number?"),
            (phone or "My number is on file.", f"Thanks. I found {name}. Which doctor would you like?"),
            (f"Please book {doctor}.", f"What day and time works for you?"),
            (when or "As soon as possible.", f"Just to confirm: booking {name} with {doctor} at {when}. Should I go ahead?"),
            ("Yes, please.", f"Done. Your appointment id is {aid}."),
        ]

    call_id = f"book-{aid.lower()}" if aid else f"book-{core._safe_id(pid)}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    data = start_call(
        call_id,
        pipeline_mode="seed" if transcript else "booking",
        channel="admin_seed",
        user_id=f"usr_{core._safe_id(pid)}",
        audio_codec="none",
        extra={
            "patient_id": pid,
            "appointment_id": aid,
            "phone": phone,
            "kind": "appointment_booking",
        },
    )
    path = core._path_for(call_id)
    with core._LOCK:
        data = core._read(path) or data
        data["patient_id"] = pid
        data["appointment_id"] = aid
        data["phone"] = phone
        interactions: list[dict[str, Any]] = []
        for i, (user_text, bot_text) in enumerate(transcript, start=1):
            ix = _empty_interaction(i, mode="text")
            ix["agent"] = "booking"
            payload = ix.setdefault("payload", {})
            payload["user_input_text"] = user_text
            payload["bot_response_text"] = bot_text
            ix["complexity_metrics"] = {
                "user_input_tokens": core._estimate_tokens(user_text),
                "bot_response_tokens": core._estimate_tokens(bot_text),
            }
            interactions.append(ix)
        data["interactions"] = interactions
        data["session_end_time"] = core._utc_iso()
        core._write(path, data)
        return data
