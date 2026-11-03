"""Speculative voice turns: held writes, scratch threads, hit and miss."""

from __future__ import annotations

import asyncio
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import agent_turn
import speculation as sp
import voice_bridge


# ── write hold ──────────────────────────────────────────────────────────────


def test_hold_is_noop_outside_speculation():
    sp.hold_side_effects()  # must not block or raise


def test_hold_waits_then_proceeds_on_confirm():
    gate = sp.SpeculationGate()
    done = threading.Event()

    def tool():
        sp.speculation_gate.set(gate)
        sp.hold_side_effects()
        done.set()

    t = threading.Thread(target=tool)
    t.start()
    assert not done.wait(0.1)  # held
    gate.confirm()
    assert done.wait(1)
    t.join()


def test_hold_refuses_write_on_abandon():
    gate = sp.SpeculationGate()
    gate.abandon()
    token = sp.speculation_gate.set(gate)
    try:
        with pytest.raises(sp.SpeculationAbandoned):
            sp.hold_side_effects()
    finally:
        sp.speculation_gate.reset(token)


def test_abandon_is_not_a_regular_exception():
    # LangGraph's tool error handling catches Exception; a refused write must
    # end the guess instead of becoming a message the model answers.
    assert not issubclass(sp.SpeculationAbandoned, Exception)


def test_gate_reaches_to_thread_workers():
    gate = sp.SpeculationGate()

    async def run():
        sp.speculation_gate.set(gate)
        return await asyncio.to_thread(sp.speculation_gate.get)

    assert asyncio.run(run()) is gate


def test_normalize_ignores_case_and_punctuation():
    assert sp.normalize_utterance("Thursday at ten.") == sp.normalize_utterance("thursday  at TEN")


# ── run_turn on a scratch thread ────────────────────────────────────────────


class FakeApp:
    """Minimal LangGraph stand-in: records invokes, replies with fixed text."""

    def __init__(self):
        self.invokes = []

    def invoke(self, payload, config):
        self.invokes.append((payload, config))
        msgs = list(payload.get("messages") or [])
        return {"messages": [*msgs, AIMessage(content="Sure, booked.")], "last_agent": "booking"}

    def get_state(self, config):
        class Snap:
            values = {"messages": [HumanMessage(content="hi")], "last_agent": "router"}

        return Snap()


def test_run_turn_seeds_history_and_defers_logs(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_turn, "_log_turn_tool_calls", lambda *a, **k: calls.append("tools"))
    app = FakeApp()
    seed = {"messages": [HumanMessage(content="earlier")], "last_agent": "router"}
    logs: list = []

    result = agent_turn.run_turn(
        app, "book me in", "t1~spec-x", call_id="c1", seed_state=seed, deferred_logs=logs
    )

    assert result.text == "Sure, booked."
    payload, config = app.invokes[0]
    assert config["configurable"]["thread_id"] == "t1~spec-x"
    assert payload["last_agent"] == "router"
    assert payload["messages"][0].content == "earlier"
    assert payload["messages"][-1] == ("user", "book me in")
    assert calls == []  # nothing logged yet
    for fn in logs:
        fn()
    assert calls == ["tools"]


# ── processor: hit and miss ─────────────────────────────────────────────────


class FakeCheckpointer:
    def __init__(self):
        self.deleted = []

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)


def _processor(monkeypatch, base="thread-A"):
    app = FakeApp()
    app.checkpointer = FakeCheckpointer()
    runs = []

    def fake_run_turn(graph_app, text, thread_id, **kwargs):
        runs.append((text, thread_id, kwargs))
        if kwargs.get("deferred_logs") is not None:
            kwargs["deferred_logs"].append(lambda: runs.append(("logged", thread_id, {})))
        return agent_turn.TurnResult(text=f"reply to {text}", agent="booking")

    monkeypatch.setattr(voice_bridge, "run_turn", fake_run_turn)
    voice_bridge._THREAD_ALIAS.pop(base, None)
    proc = voice_bridge.make_langgraph_processor(
        base, app, call_id="call-1", shared_state={}, speculation="vad",
        turn_text=sp.UserTurnText(),
    )
    return proc, app, runs


def test_speculation_hit_reuses_result_and_switches_thread(monkeypatch):
    proc, app, runs = _processor(monkeypatch)

    async def scenario():
        proc._speculate("next tuesday at ten")
        spec = proc._speculation
        assert spec is not None and spec.thread_id.startswith("thread-A~spec-")
        result = await proc._await_speculation(spec)
        return spec, result

    spec, result = asyncio.run(scenario())
    assert result.text == "reply to next tuesday at ten"
    assert runs[0][2]["seed_state"]["last_agent"] == "router"
    assert ("logged", spec.thread_id, {}) in runs  # deferred logs ran on hit
    assert proc._thread_id == spec.thread_id
    assert voice_bridge.resolve_thread_id("thread-A") == spec.thread_id
    assert app.checkpointer.deleted == []


def test_speculation_miss_discards_thread_and_logs(monkeypatch):
    proc, app, runs = _processor(monkeypatch, base="thread-B")

    async def scenario():
        proc._speculate("next tuesday")
        spec = proc._speculation
        proc._abandon(spec, reason="resumed")
        await asyncio.wait([spec.task])
        await asyncio.sleep(0)  # let the done-callback run
        return spec

    spec = asyncio.run(scenario())
    assert spec.gate.abandoned is True
    assert app.checkpointer.deleted == [spec.thread_id]
    assert not any(r[0] == "logged" for r in runs)
    assert proc._thread_id == "thread-B"
    assert voice_bridge.resolve_thread_id("thread-B") == "thread-B"


def test_same_words_do_not_restart_guess(monkeypatch):
    proc, _app, runs = _processor(monkeypatch, base="thread-C")

    async def scenario():
        proc._speculate("Hello there.")
        first = proc._speculation
        proc._speculate("hello there")
        assert proc._speculation is first
        await asyncio.wait([first.task])

    asyncio.run(scenario())
    assert len(runs) == 1


def test_no_guess_while_bot_waits_for_phone(monkeypatch):
    proc, _app, runs = _processor(monkeypatch, base="thread-D")
    import voice_bridge as vb

    monkeypatch.setattr(vb, "bot_awaiting_phone", lambda text: True)

    async def scenario():
        proc._speculate("nine eight seven")

    asyncio.run(scenario())
    assert proc._speculation is None and runs == []


# ── stats: extra LLM calls and time saved ───────────────────────────────────


def test_stats_hit_saves_head_start_or_whole_run():
    import speculation_stats as st
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    # Guess finished (0.9s) before commit (1.2s after pause): whole run saved.
    row = st.record_guess("s1", outcome="hit", text="a", started_wall=now,
                          started_at=10.0, done_at=10.9, committed_at=11.2, llm_calls=2, llm_ms=850)
    assert row["saved_ms"] == 900.0 and row["wait_after_commit_ms"] == 0.0
    # Guess still running at commit (0.4s head start): head start saved.
    row = st.record_guess("s1", outcome="hit", text="b", started_wall=now,
                          started_at=20.0, done_at=21.5, committed_at=20.4, llm_calls=1, llm_ms=1400)
    assert row["saved_ms"] == 400.0 and row["wait_after_commit_ms"] == 1100.0
    st.record_guess("s1", outcome="miss", reason="resumed", text="c", started_wall=now,
                    started_at=30.0, done_at=31.0, llm_calls=3, llm_ms=2000)

    t = st.session_totals("s1")
    assert (t["guesses"], t["hits"], t["misses"]) == (3, 2, 1)
    assert t["extra_llm_calls"] == 3 and t["extra_llm_ms"] == 2000.0
    assert t["hit_llm_calls"] == 3
    assert t["saved_ms_total"] == 1300.0 and t["saved_ms_avg_per_hit"] == 650.0
    assert t["miss_reasons"] == {"resumed": 1}
    assert st.summarize_csv("s1")["saved_ms_total"] == 1300.0


def test_gate_counts_llm_calls_through_timing_scope():
    import llm_timing

    gate = sp.SpeculationGate()
    llm_timing.set_scope("s2", speculative=True, on_call=gate.note_llm_call)
    for status in ("ok", "ok", "cancelled"):
        call, token = llm_timing.begin(llm_timing.current_scope(), node="booking", model="m", copy=1)
        call.mark_request()
        call.mark_body()
        llm_timing.end(token)
        call.finish(status)
    llm_timing.drain("s2")
    assert gate.llm_calls == 2  # a cancelled hedge copy is not a guess's call


def test_processor_records_hit_and_miss(monkeypatch):
    import speculation_stats as st

    proc, _app, _runs = _processor(monkeypatch, base="thread-S")

    async def scenario():
        proc._speculate("next tuesday")
        miss = proc._speculation
        proc._abandon(miss, reason="text_changed")
        await asyncio.wait([miss.task])
        await asyncio.sleep(0)
        proc._speculate("next tuesday at ten")
        hit = proc._speculation
        await asyncio.wait([hit.task])
        await proc._await_speculation(hit)

    asyncio.run(scenario())
    t = st.session_totals("call-1")
    assert (t["guesses"], t["hits"], t["misses"]) == (2, 1, 1)
    assert t["miss_reasons"] == {"text_changed": 1}
    assert t["saved_ms_total"] >= 0
