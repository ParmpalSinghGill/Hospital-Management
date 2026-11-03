"""Tell the voice layer when a tool starts, so it can say "Let me check…".

A tool turn takes two LLM calls plus the tool itself (~1.5–2.5 s). Saying a
short filler as soon as the model picks a tool fills that silence.

``run_turn`` sets a per-turn hook here; ``agent_invoke_config`` attaches a
LangChain callback that calls it on ``on_tool_start``. Context variables carry
the hook into LangGraph's worker threads, the same way ``set_current_call_id``
reaches the tools. Speculative turns never set a hook, so a discarded guess
stays silent.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

ToolStartHook = Callable[[str], None]

_HOOK: ContextVar[ToolStartHook | None] = ContextVar("tool_start_hook", default=None)

# Short and generic so they fit every industry profile.
_FILLERS: dict[str, str] = {
    "lookup_patient": "Let me look that up.",
    "save_patient": "One moment.",
    "list_doctors": "Let me check availability.",
    "book_appointment": "Let me book that for you.",
    "cancel_appointment": "Let me cancel that for you.",
    "reschedule_appointment": "Let me move that for you.",
    "get_prescriptions": "Let me pull that up.",
}
DEFAULT_FILLER = "Let me check."


def filler_for_tool(tool_name: str) -> str:
    return _FILLERS.get((tool_name or "").strip(), DEFAULT_FILLER)


def set_hook(hook: ToolStartHook | None):
    """Set the hook for this context; returns a token for ``reset_hook``."""
    return _HOOK.set(hook)


def reset_hook(token) -> None:
    _HOOK.reset(token)


def current_hook() -> ToolStartHook | None:
    return _HOOK.get()


def tool_start_callbacks() -> list[Any]:
    """LangChain callbacks for an agent invoke; empty when no hook is set.

    The hook is read here (in the graph node's context) and bound to the
    handler, so it still works if the tool runs in another thread.
    """
    hook = _HOOK.get()
    if hook is None:
        return []
    from langchain_core.callbacks import BaseCallbackHandler

    class _ToolStartNotifier(BaseCallbackHandler):
        def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
            name = str((serialized or {}).get("name") or kwargs.get("name") or "")
            try:
                hook(name)
            except Exception:
                pass

    return [_ToolStartNotifier()]
