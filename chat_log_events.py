"""Timeline, interrupt, and client event recording."""
from __future__ import annotations

import json
from typing import Any

import chat_log_core as core


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
        "at": core._utc_iso(),
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

    with core._LOCK:
        path = core._path_for(cid)
        data = core._read(path)
        if not data:
            from chat_log_turns import start_call

            data = start_call(cid)
            path = core._path_for(cid)
        timeline = data.setdefault("timeline", [])
        if not isinstance(timeline, list):
            timeline = []
            data["timeline"] = timeline
        timeline.append(entry)
        # Cap growth on very long calls.
        if len(timeline) > 500:
            data["timeline"] = timeline[-500:]
        data["updated_at"] = core._utc_iso()
        core._write(path, data)

        try:
            core._ensure_layout()
            line = json.dumps({"session_id": cid, **entry}, ensure_ascii=False)
            with core.TIMELINE_PATH.open("a", encoding="utf-8") as f:
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
    from chat_log_turns import (
        _empty_interaction,
        _interactions,
        append_or_update_turn,
        note_turn_meta,
        record_server_turn,
        start_call,
    )
    start_call(
        call_id,
        pipeline_mode=str(event.get("pipeline_mode") or ""),
        channel=str(event.get("channel") or "web_app"),
        user_id=str(event.get("user_id") or ""),
    )
    kind = str(event.get("type") or "").strip()
    patch: dict[str, Any] = {}
    new_turn = False
    now = core._utc_iso()

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
        sent = core._iso_from_ms(event.get("sent_at_ms")) or now
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
            with core._LOCK:
                path = core._path_for(call_id)
                data = core._read(path) or {}
                interactions = _interactions(data)
                if interactions:
                    return interactions[-1]
                return _empty_interaction(0)
        new_turn = True
        sent = (
            core._iso_from_ms(event.get("sent_at_ms"))
            or core._iso_from_ms(event.get("voice_end_at_ms"))
            or now
        )
        # Merge if server already opened this utterance.
        with core._LOCK:
            path = core._path_for(call_id)
            data = core._read(path) or {}
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
        patch["bot_text_first_shown_at"] = core._iso_from_ms(event.get("at_ms")) or now
        first_text = str(event.get("first_text") or event.get("text") or "").strip()
        if first_text:
            patch["bot_text"] = first_text
            patch["first_text"] = first_text
        if event.get("latency_ms") is not None:
            try:
                patch["first_text_latency_ms"] = float(event.get("latency_ms"))
            except (TypeError, ValueError):
                pass
    elif kind == "bot_text_complete":
        patch["bot_text_complete_at"] = core._iso_from_ms(event.get("at_ms")) or now
        if event.get("text"):
            patch["bot_text"] = event.get("text")
    elif kind == "bot_voice_first_heard":
        patch["bot_voice_first_heard_at"] = core._iso_from_ms(event.get("at_ms")) or now
        patch["mode"] = "voice_and_text"
        speech = str(event.get("first_speech") or event.get("text") or "").strip()
        if speech:
            patch["first_speech"] = speech
            if not patch.get("bot_text"):
                patch["bot_text"] = speech
        if event.get("latency_ms") is not None:
            try:
                patch["first_audio_latency_ms"] = float(event.get("latency_ms"))
            except (TypeError, ValueError):
                pass
    elif kind == "bot_voice_complete":
        patch["bot_voice_complete_at"] = core._iso_from_ms(event.get("at_ms")) or now
        patch["mode"] = "voice_and_text"
        if event.get("text"):
            patch["bot_text"] = event.get("text")
    else:
        patch.update({k: v for k, v in event.items() if k not in ("type", "call_id", "session_id")})

    ix = append_or_update_turn(call_id, patch, new_turn=new_turn)
    try:
        from llm_message_dump import record_client_turn_timing

        record_client_turn_timing(call_id, patch)
    except Exception:
        pass
    return ix
