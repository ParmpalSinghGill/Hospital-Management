"""Tests for per-LLM-call message dumps."""

from __future__ import annotations

from types import SimpleNamespace

from llm_message_dump import (
    list_llm_message_sessions,
    list_session_pairs,
    message_to_dict,
    read_session_pair,
    save_llm_pair,
)


def test_message_to_dict_tuple():
    d = message_to_dict(("user", "Hello"))
    assert d["role"] == "user"
    assert d["content"] == "Hello"


def test_save_pair_respects_setting_off(tmp_path, monkeypatch):
    import llm_message_dump as dump
    import service_settings as ss

    monkeypatch.setattr(dump, "MESSAGES_ROOT", tmp_path)
    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)
    ss.save_settings({"save_llm_messages": False})
    assert (
        save_llm_pair(
            passed_messages=[("user", "hi")],
            response=SimpleNamespace(generations=[[SimpleNamespace(text="hello", message=None)]]),
            node="general",
            session_id="sess_x",
        )
        is None
    )
    assert list(tmp_path.glob("**/Message*_passed.txt")) == []


def test_save_pair_and_list_oldest_first(tmp_path, monkeypatch):
    import llm_message_dump as dump
    import service_settings as ss

    monkeypatch.setattr(dump, "MESSAGES_ROOT", tmp_path)
    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)
    ss.save_settings({"save_llm_messages": True})
    dump._COUNTERS.clear()

    class Gen:
        def __init__(self, text):
            self.text = text
            self.message = None

    class Result:
        def __init__(self, text):
            self.generations = [[Gen(text)]]
            self.llm_output = {}

    n1 = save_llm_pair(
        passed_messages=[("system", "You are helpful."), ("user", "Hi")],
        response=Result("Hello!"),
        node="general",
        session_id="sess_a",
    )
    assert n1 == 1
    n2 = save_llm_pair(
        passed_messages=[("user", "Book please")],
        response=Result("Phone?"),
        node="booking",
        session_id="sess_a",
    )
    assert n2 == 2

    folder = tmp_path / "sess_a"
    assert (folder / "Message1_passed.txt").exists()
    assert (folder / "Message1_response.txt").exists()
    assert (folder / "Message2_passed.txt").exists()
    passed = (folder / "Message1_passed.txt").read_text(encoding="utf-8")
    assert "PASSED_TO_LLM" in passed
    assert "You are helpful." in passed
    assert "Hello!" in (folder / "Message1_response.txt").read_text(encoding="utf-8")

    pairs = list_session_pairs("sess_a")
    assert len(pairs) == 2
    pair = read_session_pair("sess_a", 1)
    assert pair and "Hi" in pair["passed"]
    assert "Hello!" in pair["response"]
    assert pair.get("stop_secs") is not None
    assert pair.get("start_secs") is not None

    sessions = list_llm_message_sessions(oldest_first=True)
    assert any(s["session_id"] == "sess_a" for s in sessions)

    from llm_message_dump import (
        delete_all_llm_messages,
        delete_session_messages,
        write_session_runtime_meta,
    )

    write_session_runtime_meta("sess_a", start_secs=0.2, stop_secs=1.5)
    pair2 = read_session_pair("sess_a", 1)
    assert pair2["stop_secs"] == 1.5
    assert pair2["start_secs"] == 0.2

    assert delete_session_messages("sess_a") is True
    assert list_session_pairs("sess_a") == []
    assert delete_session_messages("sess_a") is False

    save_llm_pair(
        passed_messages=[("user", "again")],
        response=Result("ok"),
        node="general",
        session_id="sess_b",
    )
    assert delete_all_llm_messages() >= 1
    assert list_llm_message_sessions() == []


def test_record_client_turn_timing(tmp_path, monkeypatch):
    import llm_message_dump as dump
    import service_settings as ss

    monkeypatch.setattr(dump, "MESSAGES_ROOT", tmp_path)
    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)
    ss.save_settings({"save_llm_messages": True})
    dump._COUNTERS.clear()

    class Result:
        def __init__(self, text: str):
            self.generations = [[SimpleNamespace(text=text, message=None)]]

    n = save_llm_pair(
        passed_messages=[("user", "hi")],
        response=Result("hello"),
        node="general",
        session_id="sess_timing",
    )
    assert n == 1
    dump.record_client_turn_timing(
        "sess_timing",
        {
            "bot_text_first_shown_at": "2026-07-22T12:00:00.000Z",
            "first_text": "Hello there",
            "first_text_latency_ms": 410,
            "bot_voice_first_heard_at": "2026-07-22T12:00:00.250Z",
            "first_speech": "Hello there",
            "first_audio_latency_ms": 660,
        },
    )
    pair = read_session_pair("sess_timing", 1)
    client = (pair.get("meta") or {}).get("client_timings") or {}
    assert client.get("first_text") == "Hello there"
    assert client.get("first_speech") == "Hello there"
    assert client.get("first_text_latency_ms") == 410
    assert client.get("first_audio_latency_ms") == 660
    timing_log = tmp_path / "sess_timing" / "turn_timings.jsonl"
    assert timing_log.exists()
    assert "first_text" in timing_log.read_text(encoding="utf-8")
