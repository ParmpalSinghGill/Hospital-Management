"""LLM dumps record tool DEFINITIONS as a separate request field, never as messages."""
from __future__ import annotations

import json

import llm_message_dump as lmd


def test_passed_block_keeps_message_order_and_lists_tools_separately():
    msgs = [("system", "You are the front desk."), ("human", "hi"), ("ai", "How can I help?"), ("human", "book")]
    tools = [{"type": "function", "function": {"name": "lookup_patient", "parameters": {}}},
             {"type": "function", "function": {"name": "book_appointment", "parameters": {}}}]
    out = lmd.format_messages_block(msgs, title="PASSED_TO_LLM", node="booking", n=1, tools=tools)
    order = [line for line in out.splitlines() if line.startswith("--- message ")]
    assert [o.split("role=")[1].strip(" -") for o in order] == ["system", "human", "ai", "human"]
    assert "tool_count: 2" in out
    assert "request.tools — separate field, NOT part of messages" in out
    # tools section comes after the messages, and no message row has role=tool
    assert out.index("===== TOOLS") > out.index("--- message 4/4")
    assert "role=tool" not in out
    payload = json.loads(out.split("===== EXACT PAYLOAD (JSON) =====")[1].split("===== READABLE")[0])
    assert [m["role"] for m in payload] == ["system", "human", "ai", "human"]


def test_no_tools_section_when_model_has_no_tools():
    out = lmd.format_messages_block([("system", "router"), ("human", "hi")], title="PASSED_TO_LLM", node="router", n=1)
    assert "tool_count: 0" in out and "===== TOOLS" not in out
