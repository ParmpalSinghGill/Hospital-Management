"""Compile the hospital multi-agent LangGraph."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pprint import pformat
from typing import Annotated, Any, List, Literal, NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from hospital_agents import (
    build_booking_agent,
    build_cancellation_agent,
    build_general_agent,
    build_prescription_agent,
    build_reschedule_agent,
    build_router_agent,
)
from hospital_routing import (
    _SYMPTOM_BOOKING_HINTS,
    _is_greeting_only,
    _is_short_affirmation,
    _message_plain,
    _message_role,
    _sticky_route_from_history,
    _wants_prescriptions,
)
from llm_message_dump import agent_invoke_config

CHAT_LOG_PATH: str | None = None


class GraphState(TypedDict):
    # add_messages keeps prior turns across invoke() calls (same thread_id).
    messages: Annotated[List[Any], add_messages]
    # Which specialist node produced the latest reply (for chat logs / admin).
    last_agent: NotRequired[str]


def _new_messages_only(prior: List[Any], result_messages: List[Any]) -> List[Any]:
    """Return only messages the sub-agent added (avoids duplicating history)."""
    if not result_messages:
        return []
    prior_len = len(prior or [])
    if len(result_messages) > prior_len:
        return list(result_messages[prior_len:])
    # Fallback: last message only if lengths match unexpectedly
    return list(result_messages[-1:])


def build_graph():
    general_agent = build_general_agent()
    booking_agent = build_booking_agent()
    cancellation_agent = build_cancellation_agent()
    reschedule_agent = build_reschedule_agent()
    prescription_agent = build_prescription_agent()
    router_agent = build_router_agent()

    def route_decider(state: GraphState) -> Literal["general", "booking", "cancelling", "rescheduling", "prescriptions"]:
        logging.info(f"route_decider \n{pformat(state)}")
        all_messages = state.get("messages", [])

        sticky = _sticky_route_from_history(
            all_messages,
            last_agent=str(state.get("last_agent") or ""),
        )
        if sticky:
            logging.info(f"route_decider sticky: \n{sticky}")
            return sticky

        result = router_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("router", recursion_limit=8),
        )
        logging.info(f"route_decider result \n{pformat(result['messages'][-1])}")

        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        response_text = getattr(final_message, "content", "") if final_message else ""
        response_text = str(response_text).strip().lower()

        choice: str
        if _wants_prescriptions(response_text) or response_text.strip() in (
            "prescription",
            "prescriptions",
        ):
            choice = "prescriptions"
        elif "cancelling" in response_text or "cancel" in response_text:
            choice = "cancelling"
        elif "rescheduling" in response_text or "reschedule" in response_text:
            choice = "rescheduling"
        elif "booking" in response_text or "book" in response_text:
            choice = "booking"
        else:
            choice = "general"

        # Soft greeting stays on general; medical need → booking (no menu words required).
        last_user = ""
        for m in reversed(all_messages or []):
            if _message_role(m) in ("human", "user"):
                last_user = _message_plain(m).lower()
                break
        if _is_greeting_only(last_user):
            choice = "general"
        elif choice == "general" and any(
            k in last_user for k in ("book", "appointment", "doctor", "visit", "department", *_SYMPTOM_BOOKING_HINTS)
        ):
            choice = "booking"
        elif choice == "general" and _wants_prescriptions(last_user):
            choice = "prescriptions"
        # Never treat "Family Medicine" department requests as prescriptions.
        if choice == "prescriptions" and (
            "book" in last_user or "appointment" in last_user or "department" in last_user
        ) and not _wants_prescriptions(last_user):
            choice = "booking"

        # Prefer staying on last specialist for yes/ok if router drifted.
        prev = str(state.get("last_agent") or "").strip().lower()
        if prev in ("booking", "cancelling", "rescheduling", "prescriptions") and _is_short_affirmation(last_user):
            choice = prev

        logging.info(f"route_decider final: \n{choice}")
        return choice  # type: ignore[return-value]

    def router_node(state: GraphState) -> GraphState:
        logging.info("Running node: router")
        return state

    def _append_chat_record(node: str, state_messages: List[Any], result_messages: List[Any]) -> None:
        try:
            user_text = ""
            for msg in reversed(state_messages):
                role = getattr(msg, "type", getattr(msg, "role", None)) if not isinstance(msg, tuple) else msg[0]
                content = getattr(msg, "content", None) if not isinstance(msg, tuple) else msg[1]
                if role in ("human", "user"):
                    user_text = content if isinstance(content, str) else str(content)
                    break
            assistant_text = ""
            if result_messages:
                final = result_messages[-1]
                assistant_text = getattr(final, "content", None)
                if not isinstance(assistant_text, str):
                    assistant_text = str(final)
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "node": node,
                "user": user_text,
                "assistant": assistant_text,
            }
            if CHAT_LOG_PATH:
                with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            logging.error(f"Failed to append chat record: {e}")

    def general_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: general \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = general_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("general", recursion_limit=12),
        )
        logging.info(f"Running node: general result \n{pformat(result)}")
        msgs = result.get("messages", [])
        _append_chat_record("general", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "general"}

    def booking_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: booking \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = booking_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("booking", recursion_limit=20),
        )
        msgs = result.get("messages", [])
        _append_chat_record("booking", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "booking"}

    def cancelling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: cancelling \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = cancellation_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("cancelling", recursion_limit=12),
        )
        msgs = result.get("messages", [])
        _append_chat_record("cancelling", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "cancelling"}

    def rescheduling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: rescheduling \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = reschedule_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("rescheduling", recursion_limit=16),
        )
        msgs = result.get("messages", [])
        _append_chat_record("rescheduling", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "rescheduling"}

    def prescriptions_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: prescriptions \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = prescription_agent.invoke(
            {"messages": all_messages},
            config=agent_invoke_config("prescriptions", recursion_limit=16),
        )
        msgs = result.get("messages", [])
        _append_chat_record("prescriptions", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "prescriptions"}

    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancelling", cancelling_node)
    graph.add_node("rescheduling", rescheduling_node)
    graph.add_node("prescriptions", prescriptions_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decider, {
        "general": "general",
        "booking": "booking",
        "cancelling": "cancelling",
        "rescheduling": "rescheduling",
        "prescriptions": "prescriptions",
    })
    # Let general immediately re-route in the same turn; specialists end the run
    # graph.add_edge("general", "router")
    graph.add_edge("general", END)
    graph.add_edge("booking", END)
    graph.add_edge("cancelling", END)
    graph.add_edge("rescheduling", END)
    graph.add_edge("prescriptions", END)
    
    # Add checkpointing with InMemorySaver
    return graph.compile(checkpointer=InMemorySaver())

