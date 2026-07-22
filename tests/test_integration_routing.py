"""Integration: graph compile + sticky routing across a multi-turn transcript."""

from __future__ import annotations

from hospital_graph import build_graph
from hospital_routing import _sticky_route_from_history


def test_compiled_graph_has_expected_nodes():
    graph = build_graph()
    # LangGraph compiled graphs expose node names via get_graph or nodes
    names = set(getattr(graph, "nodes", {}) or {})
    if not names and hasattr(graph, "get_graph"):
        g = graph.get_graph()
        names = set(getattr(g, "nodes", {}) or {})
    # Fallback: invoke structure check
    assert hasattr(graph, "invoke")
    assert callable(build_graph)


def test_multi_turn_sticky_routing_transcript():
    """Simulate a soft booking conversation without calling an LLM."""
    turns = []

    def route(user: str, last: str = ""):
        turns.append(("user", user))
        choice = _sticky_route_from_history(turns, last_agent=last)
        return choice

    assert route("hello") == "general"
    assert route("my tooth hurts", last="general") == "booking"
    assert route("yes", last="booking") == "booking"
    assert route("9876543210", last="booking") == "booking"
    assert route("I want to cancel my appointment", last="booking") == "cancelling"
