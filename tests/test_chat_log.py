"""Unit tests for conversation / chat log modules."""

from __future__ import annotations

import conversation_log as cl


def test_start_append_end_and_detail(isolated_chats):
    cid = "test-session-1"
    cl.start_call(cid, pipeline_mode="cli", session_id=cid, channel="cli")
    cl.append_or_update_turn(
        cid,
        {"mode": "text", "user_text": "hi", "bot_text": "hello"},
        new_turn=True,
    )
    cl.note_turn_meta(cid, agent_name="booking")
    cl.link_call_patient(cid, patient_id="PAT-0001", phone="9999999999")
    cl.end_call(cid)

    detail = cl.get_call_detail(cid)
    assert detail is not None
    assert detail["session_id"] == cid
    assert detail["patient_id"] == "PAT-0001"
    assert len(detail["turns"]) >= 1

    recent = cl.list_recent_calls(limit=5)
    assert any(r["session_id"] == cid for r in recent)
    assert cl.delete_call(cid)
    assert cl.get_call_detail(cid) is None


def test_tool_call_and_timeline(isolated_chats):
    cid = "test-session-2"
    cl.start_call(cid, pipeline_mode="cascade", session_id=cid, channel="web")
    entry = cl.record_tool_call(
        tool="list_doctors",
        arguments={"department": "Cardiology"},
        result={"ok": True, "count": 1},
        source="test",
        call_id=cid,
    )
    assert entry and entry["tool"] == "list_doctors"
    tools = cl.list_tool_calls(limit=10, source="test")
    assert any(t.get("tool") == "list_doctors" for t in tools)

    tl = cl.record_timeline_event(cid, event_type="mute", phase="tts")
    assert tl is not None
    detail = cl.get_call_detail(cid)
    assert detail is not None
    assert len(detail.get("timeline") or []) >= 1
    cl.delete_call(cid)


def test_apply_client_user_text(isolated_chats):
    cid = "test-session-3"
    cl.start_call(cid, pipeline_mode="cli", session_id=cid, channel="cli")
    cl.apply_client_event(cid, {"type": "user_text", "text": "book dentist"})
    detail = cl.get_call_detail(cid)
    assert detail and detail["turns"]
    assert "book dentist" in (detail["turns"][-1].get("user") or "")
    cl.delete_call(cid)


def test_apply_client_first_text_and_audio(isolated_chats):
    cid = "test-session-timings"
    cl.start_call(cid, pipeline_mode="cascade", session_id=cid, channel="web_app")
    cl.apply_client_event(cid, {"type": "user_voice", "text": "hello"})
    cl.apply_client_event(
        cid,
        {
            "type": "bot_text_first_shown",
            "at_ms": 1_700_000_000_100,
            "latency_ms": 420,
            "first_text": "Hi, how can I help?",
            "text": "Hi, how can I help?",
        },
    )
    cl.apply_client_event(
        cid,
        {
            "type": "bot_voice_first_heard",
            "at_ms": 1_700_000_000_350,
            "latency_ms": 670,
            "first_speech": "Hi, how can I help?",
            "text": "Hi, how can I help?",
        },
    )
    detail = cl.get_call_detail(cid)
    assert detail and detail.get("raw")
    interactions = detail["raw"].get("interactions") or []
    assert interactions
    ix = interactions[-1]
    timings = ix.get("client_timings") or {}
    assert timings.get("first_text") == "Hi, how can I help?"
    assert timings.get("first_speech") == "Hi, how can I help?"
    assert timings.get("first_text_latency_ms") == 420
    assert timings.get("first_audio_latency_ms") == 670
    assert (ix.get("timestamps") or {}).get("first_token_visible_to_user")
    assert (ix.get("timestamps") or {}).get("first_audio_heard_by_user")
    cl.delete_call(cid)


def test_record_booking_chat(isolated_chats):
    data = cl.record_booking_chat(
        patient={"patient_id": "PAT-SEED", "name": "Seed", "phone": "9000000000"},
        appointment={
            "appointment_id": "APT-SEED",
            "doctor": "Dr. Test",
            "time": "2026-12-01 10:00",
            "department": "Cardiology",
        },
    )
    assert data.get("patient_id") == "PAT-SEED"
    assert len(data.get("interactions") or []) >= 1
    cl.delete_call(data["session_id"])


def test_current_call_id_binding(isolated_chats):
    cl.set_current_call_id("abc")
    assert cl.get_current_call_id() == "abc"
    cl.set_current_call_id(None)
    assert cl.get_current_call_id() == ""


def test_list_calls_hides_seed_and_matches_name(isolated_chats, monkeypatch):
    """Seed templates stay hidden; live chats match by patient name in transcript."""
    import database as db
    from chat_log_turns import append_or_update_turn, end_call, start_call

    seed = cl.record_booking_chat(
        patient={"patient_id": "PAT-NAME", "name": "Unique Patient Zero", "phone": "9111222333"},
        appointment={
            "appointment_id": "APT-NAME",
            "doctor": "Dr. Test",
            "time": "2026-12-01 10:00",
            "department": "Cardiology",
        },
    )

    live_id = "live-name-match"
    start_call(live_id, pipeline_mode="realtime", session_id=live_id, channel="web")
    append_or_update_turn(
        live_id,
        {
            "mode": "text",
            "user_text": "book orthopedic",
            "bot_text": "क्या आप Unique Patient Zero हैं? आपका अपॉइंटमेंट बुक कर दिया है।",
        },
        new_turn=True,
    )
    end_call(live_id)

    monkeypatch.setattr(
        db,
        "_get_patient_by_id",
        lambda pid: {
            "patient_id": "PAT-NAME",
            "name": "Unique Patient Zero",
            "phone": "9111222333",
        },
    )
    monkeypatch.setattr(db, "_load_all_appointments", lambda: [])

    chats = cl.list_calls_for_patient("PAT-NAME")
    ids = [c["session_id"] for c in chats]
    assert live_id in ids
    assert seed["session_id"] not in ids

    with_seed = cl.list_calls_for_patient("PAT-NAME", include_seed=True)
    assert any(c["session_id"] == seed["session_id"] and c.get("is_seed") for c in with_seed)


def test_summarize_response_timings(isolated_chats):
    from chat_log_turns import append_or_update_turn, end_call, start_call

    start_call("avg-a", pipeline_mode="cascade", session_id="avg-a", channel="web", user_id="u1")
    append_or_update_turn(
        "avg-a",
        {
            "user_text": "hi",
            "bot_text": "hello",
            "first_text": "hello",
            "first_speech": "hello",
            "first_text_latency_ms": 400,
            "first_audio_latency_ms": 700,
            "bot_text_first_shown_at": "2026-07-22T12:00:00.000Z",
            "bot_voice_first_heard_at": "2026-07-22T12:00:00.300Z",
        },
        new_turn=True,
    )
    end_call("avg-a")

    start_call("avg-b", pipeline_mode="cascade", session_id="avg-b", channel="web", user_id="u2")
    append_or_update_turn(
        "avg-b",
        {
            "user_text": "hi",
            "bot_text": "hey",
            "first_text": "hey",
            "first_text_latency_ms": 600,
            "bot_text_first_shown_at": "2026-07-22T12:01:00.000Z",
        },
        new_turn=True,
    )
    end_call("avg-b")

    summary = cl.summarize_response_timings()
    assert summary["session_count"] >= 2
    assert summary["user_count"] >= 2
    assert summary["first_text"]["count"] == 2
    assert summary["first_text"]["avg_ms"] == 500.0
    assert summary["first_speech"]["count"] == 1
    assert summary["first_speech"]["avg_ms"] == 700.0
    assert "sessions" not in summary

    one = cl.session_response_timings("avg-a")
    assert one and one["avg_first_text_ms"] == 400.0
