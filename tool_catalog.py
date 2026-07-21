"""Machine-readable catalog of hospital appointment tools (Tools.py).

Used by HTTP discovery routes and any external integrator that needs tool
metadata without importing LangGraph or Pipecat.
"""

from __future__ import annotations

from typing import Any

from Tools import (
    book_appointment,
    cancel_appointment,
    get_prescriptions,
    list_doctors,
    lookup_patient,
    reschedule_appointment,
    save_patient,
)

# Stable order for docs and UIs.
ALL_TOOLS = (
    lookup_patient,
    save_patient,
    list_doctors,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
    get_prescriptions,
)

# Which LangGraph specialist nodes can call each tool (from Main.py).
AGENT_TOOL_MAP: dict[str, list[str]] = {
    "booking": [
        "lookup_patient",
        "save_patient",
        "list_doctors",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
    ],
    "cancelling": [
        "lookup_patient",
        "cancel_appointment",
        "book_appointment",
        "reschedule_appointment",
        "list_doctors",
    ],
    "rescheduling": [
        "lookup_patient",
        "reschedule_appointment",
        "cancel_appointment",
        "book_appointment",
        "list_doctors",
    ],
    "prescriptions": ["get_prescriptions"],
}


def _parameter_schema(tool: Any) -> dict[str, Any]:
    """JSON Schema object for tool arguments (OpenAI-compatible shape)."""
    try:
        raw = tool.get_input_schema().model_json_schema()
    except Exception:
        return {"type": "object", "properties": {}, "required": []}

    properties = dict(raw.get("properties") or {})
    # Drop pydantic/json-schema noise external clients do not need.
    for prop in properties.values():
        if isinstance(prop, dict):
            prop.pop("title", None)

    return {
        "type": "object",
        "properties": properties,
        "required": list(raw.get("required") or []),
    }


def tool_entry(tool: Any) -> dict[str, Any]:
    """One tool record for discovery APIs."""
    agents = [agent for agent, names in AGENT_TOOL_MAP.items() if tool.name in names]
    return {
        "name": tool.name,
        "description": (tool.description or "").strip(),
        "parameters": _parameter_schema(tool),
        "agents": agents,
    }


def build_tool_catalog() -> dict[str, Any]:
    """Full catalog payload returned by GET /api/tools."""
    tools = [tool_entry(t) for t in ALL_TOOLS]
    return {
        "ok": True,
        "count": len(tools),
        "tools": tools,
        "agents": AGENT_TOOL_MAP,
        "discovery": {
            "list": "/api/tools",
            "detail": "/api/tools/{name}",
            "html": "/toollist/",
        },
        "formats": ["json_schema", "openai_functions"],
    }


def get_tool_by_name(name: str) -> dict[str, Any] | None:
    wanted = (name or "").strip()
    if not wanted:
        return None
    for tool in ALL_TOOLS:
        if tool.name == wanted:
            return tool_entry(tool)
    return None
