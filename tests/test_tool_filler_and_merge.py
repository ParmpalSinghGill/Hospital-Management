"""Tool "Let me check…" filler and merging of paused speech fragments."""


import asyncio
import csv
from typing import Annotated, TypedDict

import pytest

import conversation_log as cl
import tool_progress
from chat_log_events import join_utterance


# ── tool filler hook ────────────────────────────────────────────────────


def test_filler_phrases():
    assert tool_progress.filler_for_tool("lookup_patient") == "Let me look that up."
    assert tool_progress.filler_for_tool("book_appointment") == "Let me book that for you."
    assert tool_progress.filler_for_tool("something_new") == tool_progress.DEFAULT_FILLER


def test_no_callbacks_without_hook():
    assert tool_progress.tool_start_callbacks() == []


def test_hook_fires_from_nested_agent_in_worker_thread(isolated_chats):
    """Same shape as voice: run_turn in a thread → graph node → ReAct agent → tool."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph, add_messages
    from langgraph.prebuilt import create_react_agent

    from agent_turn import run_turn
    from llm_message_dump import agent_invoke_config

    class _Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    @tool
    def lookup_patient(phone: str) -> str:
        """Look up a patient."""
        return "found"

    model = _Fake(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup_patient", "args": {"phone": "1"}, "id": "c1"}],
                ),
                AIMessage(content="Found you."),
            ]
        )
    )
    inner = create_react_agent(model, [lookup_patient])

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        last_agent: str

    def node(state):
        r = inner.invoke(
            {"messages": state["messages"]},
            config=agent_invoke_config("booking", recursion_limit=12),
        )
        return {"messages": r["messages"][len(state["messages"]):], "last_agent": "booking"}

    g = StateGraph(S)
    g.add_node("booking", node)
    g.add_edge(START, "booking")
    g.add_edge("booking", END)
    app = g.compile(checkpointer=MemorySaver())

    seen: list[str] = []

    async def go():
        return await asyncio.to_thread(
            run_turn, app, "hi", "t-filler", call_id="t-filler", on_tool_start=seen.append
        )

    result = asyncio.run(go())
    assert result.text == "Found you."
    assert seen == ["lookup_patient"]
    # Hook is cleared after the turn.
    assert tool_progress.current_hook() is None


# ── turn metrics ignore filler audio ────────────────────────────────────


def _push(obs, frame):
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    asyncio.run(
        obs.on_push_frame(
            FramePushed(
                source=None,
                destination=None,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
    )


@pytest.mark.parametrize("gap_between", [True, False])
def test_metrics_time_reply_not_filler(isolated_chats, tmp_path, monkeypatch, gap_between):
    import turn_metrics
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        TTSSpeakFrame,
        TTSStartedFrame,
        UserStoppedSpeakingFrame,
    )

    csv_path = tmp_path / "bot_messages.csv"
    monkeypatch.setattr(turn_metrics, "BOT_METRICS_CSV", csv_path)
    obs = turn_metrics.TurnMetricsObserver(session_id="s-filler", pipeline_mode="cascade")

    _push(obs, UserStoppedSpeakingFrame())
    filler = TTSSpeakFrame(text="Let me check.", append_to_context=False)
    filler.tool_filler = True
    _push(obs, filler)
    _push(obs, TTSStartedFrame())
    _push(obs, BotStartedSpeakingFrame())
    if gap_between:
        # Filler finishes before the reply exists; must not close the turn.
        _push(obs, BotStoppedSpeakingFrame())
        assert not csv_path.exists()
    _push(obs, TTSSpeakFrame(text="Found you."))
    _push(obs, TTSStartedFrame())
    if gap_between:
        _push(obs, BotStartedSpeakingFrame())
    _push(obs, BotStoppedSpeakingFrame())

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["filler_audio_heard_ms"] != ""
    assert row["first_audio_heard_ms"] != ""
    # Reply audio comes after the reply text, never before (the old bug).
    assert float(row["first_audio_heard_ms"]) >= float(row["full_text_response_ms"])
    assert float(row["first_audio_heard_ms"]) >= float(row["filler_audio_heard_ms"])


# ── paused speech fragments become one message ──────────────────────────


def test_join_utterance():
    assert join_utterance("Actually, I", "see... Yeah.") == "Actually, I see... Yeah."
    assert join_utterance("", "Hi") == "Hi"
    assert join_utterance("book at nine", "nine") == "book at nine"
    assert join_utterance("book", "book at nine") == "book at nine"


def test_voice_fragments_merge_until_bot_replies(isolated_chats):
    cid = "test-fragments"
    cl.start_call(cid, pipeline_mode="cascade", session_id=cid, channel="web_app")
    cl.apply_client_event(cid, {"type": "user_voice", "text": "I know I don't have any preference, so please"})
    cl.apply_client_event(cid, {"type": "user_voice", "text": "the best one and"})
    cl.apply_client_event(cid, {"type": "user_voice", "text": "earliest available for tomorrow."})
    cl.apply_client_event(cid, {"type": "bot_text_complete", "at_ms": 1, "text": "Dr. Foreman has 9 AM."})
    cl.apply_client_event(cid, {"type": "user_voice", "text": "Yeah. Book that."})

    ixs = cl.get_call_detail(cid)["raw"]["interactions"]
    users = [ix["payload"]["user_input_text"] for ix in ixs]
    assert users == [
        "I know I don't have any preference, so please the best one and earliest available for tomorrow.",
        "Yeah. Book that.",
    ]
    cl.delete_call(cid)
