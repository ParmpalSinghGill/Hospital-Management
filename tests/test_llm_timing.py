"""Per-LLM-call T1–T4 timing and the per-turn summary."""

from __future__ import annotations

import csv

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

import llm_timing


@pytest.fixture(autouse=True)
def _csv(tmp_path, monkeypatch):
    path = tmp_path / "llm_calls.csv"
    monkeypatch.setattr(llm_timing, "LLM_CALLS_CSV", path)
    llm_timing._PENDING.clear()
    return path


def _call(session="s1", *, t1, t2, t3, t4, spec=False, node="booking"):
    c = llm_timing.LLMCallTiming(session_id=session, speculative=spec, node=node, model="m")
    c.t0, c.t1, c.t2, c.t3, c.t4 = t1, t1, t2, t3, t4
    return c


def test_chunks_without_text_do_not_mark_tokens():
    c = llm_timing.LLMCallTiming()
    c.mark_chunk(ChatGenerationChunk(message=AIMessageChunk(content="")))
    assert c.t3 is None
    c.mark_chunk(ChatGenerationChunk(message=AIMessageChunk(content="Hi")))
    first = c.t3
    c.mark_chunk(ChatGenerationChunk(message=AIMessageChunk(content=" there")))
    assert c.t3 == first and c.t4 >= first and c.chunks == 2


def test_tool_call_chunk_counts_as_token():
    c = llm_timing.LLMCallTiming()
    chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "list_doctors", "args": "", "id": "1", "index": 0}]
    )
    c.mark_chunk(ChatGenerationChunk(message=chunk))
    assert c.t3 is not None


def test_finish_writes_csv_and_queues_only_ok_calls(_csv):
    _call(t1=0.0, t2=0.2, t3=0.8, t4=1.0).finish("ok")
    _call(t1=0.0, t2=None, t3=None, t4=None).finish("cancelled")
    rows = list(csv.DictReader(_csv.open()))
    assert [r["status"] for r in rows] == ["ok", "cancelled"]
    assert rows[0]["wait_ms"] == "200.0" and rows[0]["first_token_ms"] == "600.0"
    assert len(llm_timing.drain("s1")) == 1
    assert llm_timing.drain("s1") == []


def test_summary_spans_first_and_last_call():
    calls = [
        _call(t1=0.01, t2=0.25, t3=0.9, t4=1.0),  # tool round
        _call(t1=1.1, t2=1.35, t3=2.0, t4=2.2),  # final answer
    ]
    s = llm_timing.summarize_turn(calls, t0=0.0)
    assert s["llm_calls"] == "2"
    assert s["t1_llm_request_ms"] == "10.0"
    assert s["t2_llm_first_byte_ms"] == "250.0"
    assert s["t3_llm_first_token_ms"] == "900.0"
    assert s["t4_llm_last_token_ms"] == "2200.0"
    assert s["between_llm_calls_ms"] == "100.0"  # tool execution gap
    assert s["llm_nodes"] == "booking+booking"


def test_summary_ignores_discarded_guess():
    guess = _call(t1=-0.3, t2=-0.1, t3=0.2, t4=0.4, spec=True)
    real = _call(t1=0.01, t2=0.2, t3=0.7, t4=0.9)
    s = llm_timing.summarize_turn([guess, real], t0=0.0)
    assert s["llm_calls"] == "1" and s["t1_llm_request_ms"] == "10.0"


def test_summary_uses_guess_when_it_was_the_answer():
    guess = _call(t1=-0.3, t2=-0.1, t3=0.2, t4=0.4, spec=True)
    s = llm_timing.summarize_turn([guess], t0=0.0)
    assert s["t1_llm_request_ms"] == "-300.0"  # started before the turn ended


def test_empty_summary_has_all_columns():
    assert set(llm_timing.summarize_turn([], 0.0)) == set(llm_timing.TURN_COLUMNS)
