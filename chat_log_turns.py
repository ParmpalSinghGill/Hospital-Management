"""Session lifecycle and turn recording."""
from __future__ import annotations

from typing import Any

from chat_log_core import (
    _LOCK,
    _OPEN,
    _ensure_layout,
    _estimate_tokens,
    _iso_from_ms,
    _path_for,
    _read,
    _safe_id,
    _utc_iso,
    _write,
    set_current_call_id,
)

_PENDING_TURN_META: dict[str, dict[str, Any]] = {}


def _channel_label(channel: str) -> str:
    c = (channel or "web").strip().lower()
    if c in ("web", "web_app", "browser"):
        return "web_app"
    if c in ("cli", "text"):
        return "cli"
    if c in ("voice", "webrtc"):
        return "web_app"
    if c in ("telegram", "tg"):
        return "telegram"
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

        timings = ix.setdefault("client_timings", {})
        if not isinstance(timings, dict):
            timings = {}
            ix["client_timings"] = timings
        if patch.get("first_text"):
            timings["first_text"] = str(patch["first_text"])
        if patch.get("first_speech"):
            timings["first_speech"] = str(patch["first_speech"])
        if patch.get("first_text_latency_ms") is not None:
            try:
                timings["first_text_latency_ms"] = round(float(patch["first_text_latency_ms"]), 1)
            except (TypeError, ValueError):
                pass
        if patch.get("first_audio_latency_ms") is not None:
            try:
                timings["first_audio_latency_ms"] = round(float(patch["first_audio_latency_ms"]), 1)
            except (TypeError, ValueError):
                pass
        if patch.get("bot_text_first_shown_at"):
            timings["first_text_at"] = patch["bot_text_first_shown_at"]
        if patch.get("bot_voice_first_heard_at"):
            timings["first_audio_at"] = patch["bot_voice_first_heard_at"]

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

    return append_or_update_turn(
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
