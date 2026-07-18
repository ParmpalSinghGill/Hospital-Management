"""Per-session conversation JSON logs in the product schema.

Saved under ``chats/`` as::

    {
      "session_id": "...",
      "user_id": "...",
      "session_start_time": "...Z",
      "session_end_time": "...Z",
      "device_info": {"channel": "web_app", "audio_codec": "opus"},
      "interactions": [ ... ]
    }
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
CHATS_DIR = Path(os.environ.get("CHAT_LOGS_DIR", _ROOT / "chats"))
_LOCK = threading.RLock()
_OPEN: dict[str, Path] = {}
_CURRENT_CALL_ID = threading.local()


def set_current_call_id(call_id: str | None) -> None:
    """Bind the active voice/CLI call id so tools can tag the open chat log."""
    _CURRENT_CALL_ID.value = (call_id or "").strip()


def get_current_call_id() -> str:
    return str(getattr(_CURRENT_CALL_ID, "value", "") or "").strip()


def _utc_iso(dt: datetime | None = None) -> str:
    """UTC ISO-8601 with millisecond precision and trailing Z."""
    d = dt or datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone(timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def _iso_from_ms(ms: float | int | None) -> str | None:
    if ms is None:
        return None
    try:
        return _utc_iso(datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc))
    except (TypeError, ValueError, OSError):
        return None


def _safe_id(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in (value or "").strip())
    return cleaned[:80] or "unknown"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words + punctuation); good enough for logs."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    parts = re.findall(r"\w+|[^\w\s]", cleaned, flags=re.UNICODE)
    return max(1, len(parts))


def _path_for(session_id: str) -> Path:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    if session_id in _OPEN:
        return _OPEN[session_id]
    sid = _safe_id(session_id)
    matches = sorted(CHATS_DIR.glob(f"*_{sid}.json")) + sorted(
        CHATS_DIR.glob(f"sess_*_{sid}.json")
    )
    if matches:
        path = matches[-1]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = CHATS_DIR / f"sess-{stamp}_{sid}.json"
    _OPEN[session_id] = path
    return path


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _channel_label(channel: str) -> str:
    c = (channel or "web").strip().lower()
    if c in ("web", "web_app", "browser"):
        return "web_app"
    if c in ("cli", "text"):
        return "cli"
    if c in ("voice", "webrtc"):
        return "web_app"
    return c or "web_app"


def _empty_interaction(turn_number: int, mode: str = "text") -> dict[str, Any]:
    return {
        "turn_number": turn_number,
        "mode": mode,
        "agent": "",
        "payload": {
            "user_input_text": "",
            "bot_response_text": "",
        },
        "complexity_metrics": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "timestamps": {},
    }


_PENDING_TURN_META: dict[str, dict[str, Any]] = {}


def note_turn_meta(
    call_id: str,
    *,
    agent_name: str = "",
    patient_id: str = "",
    phone: str = "",
    appointment_id: str = "",
) -> None:
    """Stash per-turn metadata (agent / patient) until record_server_turn writes it."""
    cid = (call_id or "").strip()
    if not cid:
        return
    with _LOCK:
        meta = _PENDING_TURN_META.setdefault(cid, {})
        if agent_name:
            meta["agent_name"] = str(agent_name)
        if patient_id:
            meta["patient_id"] = str(patient_id)
        if phone:
            meta["phone"] = str(phone)
        if appointment_id:
            meta["appointment_id"] = str(appointment_id)


def link_call_patient(
    call_id: str,
    *,
    patient_id: str = "",
    phone: str = "",
    appointment_id: str = "",
) -> None:
    """Tag an open/live call log with patient identity so admin can list it."""
    cid = (call_id or "").strip()
    if not cid:
        return
    note_turn_meta(
        cid,
        patient_id=patient_id,
        phone=phone,
        appointment_id=appointment_id,
    )
    with _LOCK:
        path = _path_for(cid)
        data = _read(path)
        if not data:
            data = start_call(cid)
            path = _path_for(cid)
        meta = data.setdefault("meta", {})
        if patient_id:
            data["patient_id"] = str(patient_id)
            meta["patient_id"] = str(patient_id)
        if phone:
            meta["phone"] = str(phone)
            data["phone"] = str(phone)
        if appointment_id:
            data["appointment_id"] = str(appointment_id)
            meta["appointment_id"] = str(appointment_id)
        data["updated_at"] = _utc_iso()
        _write(path, data)


def _interactions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize older ``turns`` logs into ``interactions`` when needed."""
    if isinstance(data.get("interactions"), list):
        return list(data["interactions"])
    legacy = data.get("turns") or []
    out: list[dict[str, Any]] = []
    for i, t in enumerate(legacy, start=1):
        if not isinstance(t, dict):
            continue
        mode = "voice_and_text" if t.get("bot_voice_first_heard_at") or t.get("input_type") == "voice" else "text"
        ts: dict[str, Any] = {}
        if t.get("user_sent_at"):
            ts["user_sent_time"] = t["user_sent_at"]
        if t.get("bot_text_first_shown_at") or t.get("bot_text_first_token_at"):
            ts["first_token_visible_to_user"] = (
                t.get("bot_text_first_shown_at") or t.get("bot_text_first_token_at")
            )
        if t.get("bot_text_complete_at"):
            ts["full_response_visible_to_user"] = t["bot_text_complete_at"]
        if t.get("bot_voice_first_heard_at"):
            ts["first_audio_heard_by_user"] = t["bot_voice_first_heard_at"]
        if t.get("bot_voice_complete_at"):
            ts["audio_playback_ended"] = t["bot_voice_complete_at"]
        user_text = t.get("user_text") or ""
        bot_text = t.get("bot_text") or ""
        out.append(
            {
                "turn_number": t.get("turn") or i,
                "mode": mode,
                "payload": {
                    "user_input_text": user_text,
                    "bot_response_text": bot_text,
                },
                "complexity_metrics": {
                    "input_tokens": _estimate_tokens(user_text),
                    "output_tokens": _estimate_tokens(bot_text),
                },
                "timestamps": ts,
            }
        )
    return out


def _refresh_interaction(ix: dict[str, Any]) -> None:
    payload = ix.setdefault("payload", {})
    user_text = str(payload.get("user_input_text") or "")
    bot_text = str(payload.get("bot_response_text") or "")
    ix["complexity_metrics"] = {
        "input_tokens": _estimate_tokens(user_text),
        "output_tokens": _estimate_tokens(bot_text),
    }
    ts = ix.get("timestamps") or {}
    has_audio = bool(ts.get("first_audio_heard_by_user") or ts.get("audio_playback_ended"))
    # Prefer explicit mode from voice input, else derive
    if has_audio or ix.get("mode") == "voice_and_text":
        ix["mode"] = "voice_and_text"
    else:
        ix["mode"] = "text"


def start_call(
    call_id: str,
    *,
    pipeline_mode: str = "",
    session_id: str = "",
    channel: str = "web",
    user_id: str = "",
    audio_codec: str = "opus",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or reopen a session log. ``call_id`` is always the file key."""
    # Never let a separate WebRTC UUID become a second log file.
    sid = call_id or session_id
    with _LOCK:
        path = _path_for(sid)
        data = _read(path)
        if not data or "interactions" not in data and "turns" not in data and "session_id" not in data:
            data = {
                "session_id": sid,
                "user_id": user_id or "usr_anonymous",
                "session_start_time": _utc_iso(),
                "session_end_time": None,
                "device_info": {
                    "channel": _channel_label(channel),
                    "audio_codec": audio_codec or "opus",
                },
                "pipeline_mode": pipeline_mode or "",
                "interactions": [],
            }
            if extra:
                data["meta"] = extra
        else:
            data["interactions"] = _interactions(data)
            data.pop("turns", None)
            if user_id and data.get("user_id") in (None, "", "usr_anonymous"):
                data["user_id"] = user_id
            if pipeline_mode and not data.get("pipeline_mode"):
                data["pipeline_mode"] = pipeline_mode
            device = data.setdefault("device_info", {})
            if channel:
                device["channel"] = _channel_label(channel)
            if audio_codec and not device.get("audio_codec"):
                device["audio_codec"] = audio_codec
            if extra:
                meta = data.setdefault("meta", {})
                meta.update(extra)
        _write(path, data)
        return data


def end_call(call_id: str) -> dict[str, Any] | None:
    with _LOCK:
        path = _path_for(call_id)
        data = _read(path)
        if not data:
            return None
        data["interactions"] = _interactions(data)
        data.pop("turns", None)
        data["session_end_time"] = _utc_iso()
        _write(path, data)
        _OPEN.pop(call_id, None)
        return data


def _set_timestamp(ix: dict[str, Any], key: str, value: str | None, *, preserve: bool = True) -> None:
    if not value:
        return
    ts = ix.setdefault("timestamps", {})
    if preserve and ts.get(key):
        return
    ts[key] = value


def _set_text(payload: dict[str, Any], key: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value)
    existing = str(payload.get(key) or "")
    if not existing or len(text) >= len(existing):
        payload[key] = text


def append_or_update_turn(
    call_id: str, patch: dict[str, Any], *, new_turn: bool = False
) -> dict[str, Any]:
    """Append a new interaction or merge ``patch`` into the latest one.

    Accepts either the product schema fields or the older flat keys used by
    the client/server event pipeline.
    """
    with _LOCK:
        path = _path_for(call_id)
        data = _read(path)
        if not data:
            data = start_call(call_id)
            path = _path_for(call_id)
        data["interactions"] = _interactions(data)
        data.pop("turns", None)
        interactions: list[dict[str, Any]] = list(data.get("interactions") or [])

        create = new_turn or not interactions
        if create:
            mode = patch.get("mode") or (
                "voice_and_text"
                if patch.get("input_type") == "voice"
                else "text"
            )
            ix = _empty_interaction(len(interactions) + 1, mode=str(mode))
            interactions.append(ix)
        else:
            ix = interactions[-1]

        payload = ix.setdefault("payload", {})

        # Product-schema style patch
        if "payload" in patch and isinstance(patch["payload"], dict):
            _set_text(payload, "user_input_text", patch["payload"].get("user_input_text"))
            _set_text(payload, "bot_response_text", patch["payload"].get("bot_response_text"))
        if "timestamps" in patch and isinstance(patch["timestamps"], dict):
            for k, v in patch["timestamps"].items():
                _set_timestamp(ix, k, v if isinstance(v, str) else None)
        if "mode" in patch and patch["mode"]:
            ix["mode"] = patch["mode"]
        if "complexity_metrics" in patch and isinstance(patch["complexity_metrics"], dict):
            ix["complexity_metrics"] = {
                **(ix.get("complexity_metrics") or {}),
                **patch["complexity_metrics"],
            }

        # Legacy / event pipeline keys
        _set_text(payload, "user_input_text", patch.get("user_text"))
        _set_text(payload, "bot_response_text", patch.get("bot_text"))
        if patch.get("input_type") == "voice":
            ix["mode"] = "voice_and_text"

        _set_timestamp(ix, "user_sent_time", patch.get("user_sent_at"))
        _set_timestamp(ix, "bot_received_time", patch.get("bot_received_at"))
        _set_timestamp(
            ix,
            "first_token_visible_to_user",
            patch.get("bot_text_first_shown_at") or patch.get("bot_text_first_token_at"),
        )
        _set_timestamp(
            ix, "full_response_visible_to_user", patch.get("bot_text_complete_at")
        )
        _set_timestamp(
            ix, "first_audio_heard_by_user", patch.get("bot_voice_first_heard_at")
        )
        _set_timestamp(ix, "audio_playback_ended", patch.get("bot_voice_complete_at"))

        if patch.get("server") and isinstance(patch["server"], dict):
            ix["server"] = {**(ix.get("server") or {}), **patch["server"]}

        agent_name = patch.get("agent_name") or patch.get("agent")
        if agent_name:
            ix["agent"] = str(agent_name)

        _refresh_interaction(ix)
        data["interactions"] = interactions
        data["updated_at"] = _utc_iso()
        _write(path, data)
        return ix


def record_server_turn(
    call_id: str,
    *,
    pipeline_mode: str,
    input_type: str,
    user_text: str,
    bot_text: str = "",
    user_sent_at: str | None = None,
    bot_text_first_token_at: str | None = None,
    bot_text_complete_at: str | None = None,
    bot_voice_first_heard_at: str | None = None,
    bot_voice_complete_at: str | None = None,
    server_extra: dict[str, Any] | None = None,
    agent_name: str = "",
) -> None:
    start_call(call_id, pipeline_mode=pipeline_mode, channel="voice")
    with _LOCK:
        pending = dict(_PENDING_TURN_META.pop(call_id, {}) or {})
    agent = (agent_name or pending.get("agent_name") or "").strip()
    if pending.get("patient_id") or pending.get("phone"):
        link_call_patient(
            call_id,
            patient_id=str(pending.get("patient_id") or ""),
            phone=str(pending.get("phone") or ""),
            appointment_id=str(pending.get("appointment_id") or ""),
        )
    with _LOCK:
        path = _path_for(call_id)
        data = _read(path) or {}
        interactions = _interactions(data)
        new_turn = True
        incoming = (user_text or "").strip()
        incoming_bot = (bot_text or "").strip()
        if not incoming and not incoming_bot:
            return
        if interactions:
            last = interactions[-1]
            last_user = str((last.get("payload") or {}).get("user_input_text") or "").strip()
            last_bot = str((last.get("payload") or {}).get("bot_response_text") or "").strip()
            # Empty user text → always merge into latest (never spawn blank turns).
            if not incoming:
                new_turn = False
            elif last_user == incoming:
                new_turn = False
            elif last_user and not last_bot:
                new_turn = False
            elif not last_user:
                new_turn = False
            # Same bot reply right after a filled turn → duplicate pipeline event.
            elif incoming_bot and last_bot and incoming_bot == last_bot and not incoming:
                new_turn = False

    # Normalize Z timestamps if server passed offset form
    def _norm(v: str | None) -> str | None:
        if not v:
            return None
        if v.endswith("Z"):
            return v
        try:
            d = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return _utc_iso(d)
        except ValueError:
            return v

    append_or_update_turn(
        call_id,
        {
            "input_type": input_type,
            "user_text": user_text,
            "bot_text": bot_text,
            "agent_name": agent,
            "user_sent_at": _norm(user_sent_at),
            "bot_received_at": _norm(user_sent_at),
            "bot_text_first_token_at": _norm(bot_text_first_token_at),
            "bot_text_complete_at": _norm(bot_text_complete_at),
            "bot_voice_first_heard_at": _norm(bot_voice_first_heard_at),
            "bot_voice_complete_at": _norm(bot_voice_complete_at),
            "server": server_extra or {},
        },
        new_turn=new_turn,
    )


    return append_or_update_turn(call_id, patch, new_turn=new_turn)


def record_timeline_event(
    call_id: str,
    *,
    event_type: str,
    user_text: str = "",
    bot_text: str = "",
    strategy: str = "",
    phase: str = "",
    source: str = "server",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append a timestamped diagnostic event (interrupts, mute, barge-in, etc.).

    Saved on the call JSON under ``timeline`` and mirrored to
    ``chats/timeline.jsonl`` for quick grepping across sessions.
    """
    cid = (call_id or "").strip()
    etype = (event_type or "").strip()
    if not cid or not etype:
        return None

    entry: dict[str, Any] = {
        "at": _utc_iso(),
        "type": etype,
        "source": source or "server",
    }
    if strategy:
        entry["strategy"] = str(strategy)
    if phase:
        entry["phase"] = str(phase)
    if user_text:
        entry["user_text"] = str(user_text)[:500]
    if bot_text:
        entry["bot_text"] = str(bot_text)[:800]
    if extra:
        for k, v in extra.items():
            if v is None or k in entry:
                continue
            entry[k] = v

    with _LOCK:
        path = _path_for(cid)
        data = _read(path)
        if not data:
            data = start_call(cid)
            path = _path_for(cid)
        timeline = data.setdefault("timeline", [])
        if not isinstance(timeline, list):
            timeline = []
            data["timeline"] = timeline
        timeline.append(entry)
        # Cap growth on very long calls.
        if len(timeline) > 500:
            data["timeline"] = timeline[-500:]
        data["updated_at"] = _utc_iso()
        _write(path, data)

        try:
            CHATS_DIR.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"session_id": cid, **entry}, ensure_ascii=False)
            with (CHATS_DIR / "timeline.jsonl").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    return entry


def record_interrupt(
    call_id: str,
    *,
    reason: str = "interruption",
    user_text: str = "",
    bot_text: str = "",
    strategy: str = "",
    phase: str = "",
    source: str = "server",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convenience wrapper for interrupt / barge-in timeline events."""
    return record_timeline_event(
        call_id,
        event_type=reason or "interruption",
        user_text=user_text,
        bot_text=bot_text,
        strategy=strategy,
        phase=phase,
        source=source,
        extra=extra,
    )


def apply_client_event(call_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Merge a browser-side timing/conversation event into the latest interaction."""
    start_call(
        call_id,
        pipeline_mode=str(event.get("pipeline_mode") or ""),
        channel=str(event.get("channel") or "web_app"),
        user_id=str(event.get("user_id") or ""),
    )
    kind = str(event.get("type") or "").strip()
    patch: dict[str, Any] = {}
    new_turn = False
    now = _utc_iso()

    # Diagnostic timeline events (do not create a chat turn).
    if kind in (
        "interrupt",
        "barge_in_while_bot_speaking",
        "user_started_speaking",
        "user_stopped_speaking",
        "bot_started_speaking",
        "bot_stopped_speaking",
        "timeline",
    ):
        entry = record_timeline_event(
            call_id,
            event_type=kind if kind != "timeline" else str(event.get("event_type") or "timeline"),
            user_text=str(event.get("user_text") or event.get("text") or ""),
            bot_text=str(event.get("bot_text") or ""),
            strategy=str(event.get("strategy") or ""),
            phase=str(event.get("phase") or ""),
            source="client",
            extra={
                "client_at_ms": event.get("at_ms"),
                "bot_speaking": event.get("bot_speaking"),
            },
        )
        return entry or {}

    if kind == "user_text":
        new_turn = True
        sent = _iso_from_ms(event.get("sent_at_ms")) or now
        patch.update(
            {
                "input_type": "text",
                "mode": "text",
                "user_text": event.get("text") or "",
                "user_sent_at": sent,
                "bot_received_at": sent,
            }
        )
    elif kind == "user_voice":
        text = str(event.get("text") or "").strip()
        if not text:
            with _LOCK:
                path = _path_for(call_id)
                data = _read(path) or {}
                interactions = _interactions(data)
                if interactions:
                    return interactions[-1]
                return _empty_interaction(0)
        new_turn = True
        sent = (
            _iso_from_ms(event.get("sent_at_ms"))
            or _iso_from_ms(event.get("voice_end_at_ms"))
            or now
        )
        # Merge if server already opened this utterance.
        with _LOCK:
            path = _path_for(call_id)
            data = _read(path) or {}
            interactions = _interactions(data)
            if interactions:
                last_user = str(
                    ((interactions[-1].get("payload") or {}).get("user_input_text") or "")
                ).strip()
                if last_user == text:
                    new_turn = False
        patch.update(
            {
                "input_type": "voice",
                "mode": "voice_and_text",
                "user_text": text,
                "user_sent_at": sent,
                "bot_received_at": sent,
            }
        )
    elif kind == "bot_text_first_shown":
        patch["bot_text_first_shown_at"] = _iso_from_ms(event.get("at_ms")) or now
        if event.get("text"):
            patch["bot_text"] = event.get("text")
    elif kind == "bot_text_complete":
        patch["bot_text_complete_at"] = _iso_from_ms(event.get("at_ms")) or now
        if event.get("text"):
            patch["bot_text"] = event.get("text")
    elif kind == "bot_voice_first_heard":
        patch["bot_voice_first_heard_at"] = _iso_from_ms(event.get("at_ms")) or now
        patch["mode"] = "voice_and_text"
    elif kind == "bot_voice_complete":
        patch["bot_voice_complete_at"] = _iso_from_ms(event.get("at_ms")) or now
        patch["mode"] = "voice_and_text"
        if event.get("text"):
            patch["bot_text"] = event.get("text")
    else:
        patch.update({k: v for k, v in event.items() if k not in ("type", "call_id", "session_id")})

    return append_or_update_turn(call_id, patch, new_turn=new_turn)


def list_recent_calls(limit: int = 20) -> list[dict[str, Any]]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        list(CHATS_DIR.glob("sess-*.json")) + list(CHATS_DIR.glob("call-*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for path in files[:limit]:
        data = _read(path)
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
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        list(CHATS_DIR.glob("sess-*.json")) + list(CHATS_DIR.glob("call-*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


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


def _chat_matches_patient(
    data: dict[str, Any],
    *,
    patient_id: str,
    phone_digits: str,
    appointment_ids: set[str],
) -> bool:
    """Match tagged calls, or live calls where this patient's phone was confirmed in-chat."""
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    chat_pid = str(data.get("patient_id") or meta.get("patient_id") or "").strip().upper()
    if patient_id and chat_pid == patient_id.upper():
        return True

    aid = str(data.get("appointment_id") or meta.get("appointment_id") or "").strip().upper()
    if aid and aid in appointment_ids:
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
        blob_parts: list[str] = []
        for ix in _interactions(data):
            payload = ix.get("payload") if isinstance(ix.get("payload"), dict) else {}
            blob_parts.append(str(payload.get("user_input_text") or ""))
            blob_parts.append(str(payload.get("bot_response_text") or ""))
        blob = " ".join(blob_parts)
        blob_digits = _normalize_digits(blob)
        if phone_digits in blob_digits:
            low = blob.lower()
            if any(
                marker in low
                for marker in (
                    "found you",
                    "look you up",
                    "patient id",
                    "is that correct",
                    "phone number is",
                    "on file",
                )
            ):
                return True
    return False


def list_calls_for_patient(patient_id: str) -> list[dict[str, Any]]:
    """Return chat sessions for this patient (tagged or phone-confirmed), newest first.

    Prefer patient_id / appointment_id / stored phone on the call log (set when
    lookup_patient succeeds after the number is confirmed). Also includes older
    live calls where that phone was confirmed in the transcript but not tagged yet.
    """
    pid = (patient_id or "").strip()
    if not pid:
        return []

    phone_digits = ""
    appointment_ids: set[str] = set()
    try:
        from database import _get_patient_by_id, _load_all_appointments, _normalize_phone

        patient = _get_patient_by_id(pid)
        if patient:
            phone_digits = _normalize_phone(str(patient.get("phone") or ""))
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
        data = _read(path)
        if not data:
            continue
        if not _chat_matches_patient(
            data,
            patient_id=pid,
            phone_digits=phone_digits,
            appointment_ids=appointment_ids,
        ):
            continue
        sid = str(data.get("session_id") or data.get("call_id") or path.name)
        if sid in seen:
            continue
        seen.add(sid)
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        interactions = _interactions(data)
        kind = str(meta.get("kind") or "").strip()
        pipeline = str(data.get("pipeline_mode") or "").strip()
        if not kind:
            if sid.lower().startswith("book-") or "book-apt-" in path.name:
                kind = "appointment_booking"
            elif pipeline in ("cascade", "realtime", "cli"):
                kind = f"live:{pipeline}"
            else:
                kind = pipeline or "live"
        # Surface intent from transcript for admin scanning.
        blob = " ".join(
            str((ix.get("payload") or {}).get("user_input_text") or "")
            + " "
            + str((ix.get("payload") or {}).get("bot_response_text") or "")
            for ix in interactions
        ).lower()
        topic = ""
        if "reschedule" in blob:
            topic = "reschedule"
        elif "cancel" in blob:
            topic = "cancel"
        elif "book" in blob or "appointment" in blob:
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
                    )
                ][-40:],
            }
        )

    out.sort(
        key=lambda r: str(
            r.get("session_start_time")
            or r.get("session_end_time")
            or r.get("file")
            or ""
        ),
        reverse=True,
    )
    return out


def delete_call(session_id: str) -> bool:
    """Delete a chat log file by session_id (or matching file name). Returns True if removed."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    safe = _safe_id(sid)
    with _LOCK:
        removed = False
        # Close open handle if this session is active in-process.
        for key in list(_OPEN.keys()):
            if key == sid or _safe_id(key) == safe:
                _OPEN.pop(key, None)
        for path in list(_iter_chat_files()):
            data = _read(path)
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
        data = _read(path)
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

    call_id = f"book-{aid.lower()}" if aid else f"book-{_safe_id(pid)}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    data = start_call(
        call_id,
        pipeline_mode="seed" if transcript else "booking",
        channel="admin_seed",
        user_id=f"usr_{_safe_id(pid)}",
        audio_codec="none",
        extra={
            "patient_id": pid,
            "appointment_id": aid,
            "phone": phone,
            "kind": "appointment_booking",
        },
    )
    path = _path_for(call_id)
    with _LOCK:
        data = _read(path) or data
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
                "user_input_tokens": _estimate_tokens(user_text),
                "bot_response_tokens": _estimate_tokens(bot_text),
            }
            interactions.append(ix)
        data["interactions"] = interactions
        data["session_end_time"] = _utc_iso()
        _write(path, data)
        return data

