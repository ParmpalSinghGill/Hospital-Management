"""Interim AI speech next to tool calls is kept; leaks and stale turns are not."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_turn import extract_assistant_text


def _tc(name="lookup_patient"):
    return [{"name": name, "args": {}, "id": "call_1", "type": "tool_call"}]


def test_interim_acknowledgment_is_prepended_to_final():
    msgs = [
        HumanMessage("i have pain in knee"),
        AIMessage("Sorry to hear about the knee pain. Let me find you a doctor for that.", tool_calls=_tc()),
        ToolMessage(content="{}", tool_call_id="call_1"),
        AIMessage("Would you like to book an appointment with our Orthopedics department?"),
    ]
    assert extract_assistant_text({"messages": msgs}) == (
        "Sorry to hear about the knee pain. Let me find you a doctor for that. "
        "Would you like to book an appointment with our Orthopedics department?"
    )


def test_plain_final_reply_unchanged():
    msgs = [HumanMessage("hello"), AIMessage("Hi, how can I help you today?")]
    assert extract_assistant_text({"messages": msgs}) == "Hi, how can I help you today?"


def test_duplicate_interim_not_repeated():
    msgs = [
        HumanMessage("book me"),
        AIMessage("One moment.", tool_calls=_tc()),
        ToolMessage(content="{}", tool_call_id="call_1"),
        AIMessage("One moment."),
    ]
    assert extract_assistant_text({"messages": msgs}) == "One moment."


def test_no_final_message_still_returns_empty_for_recovery():
    msgs = [HumanMessage("book me"), AIMessage("Checking that now.", tool_calls=_tc())]
    assert extract_assistant_text({"messages": msgs}) == ""


def test_stale_previous_turn_not_included():
    msgs = [
        HumanMessage("hello"),
        AIMessage("Hi, how can I help you today?"),
        HumanMessage("i have knee pain"),
        AIMessage("Sorry about the knee pain. What's your phone number?"),
    ]
    assert extract_assistant_text({"messages": msgs}) == "Sorry about the knee pain. What's your phone number?"
