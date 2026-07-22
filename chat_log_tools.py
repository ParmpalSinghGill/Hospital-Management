"""Tool-call audit log helpers."""
from __future__ import annotations

import json
from typing import Any

from chat_log_core import (
    _LOCK,
    _estimate_tokens,
    _read_tool_call_store,
    _utc_iso,
    _write_tool_call_store,
    get_current_call_id,
)


def list_tool_calls(*, limit: int = 50, source: str = "") -> list[dict[str, Any]]:
    """Return recent tool calls from ``chats/tool_call.json`` (newest last)."""
    with _LOCK:
        store = _read_tool_call_store()
        rows = [r for r in (store.get("tool_calls") or []) if isinstance(r, dict)]
        src = (source or "").strip().lower()
        if src:
            rows = [r for r in rows if str(r.get("source") or "").lower() == src]
        limit = max(1, min(int(limit or 50), 500))
        return rows[-limit:]

def _truncate_tool_payload(value: Any, *, limit: int = 4000) -> Any:
    """Keep tool args/results loggable without huge blobs."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
    return text[: limit - 1] + "…"

def record_tool_call(
    *,
    tool: str,
    arguments: dict[str, Any] | None = None,
    result: Any = None,
    source: str = "server",
    call_id: str | None = None,
    agent: str = "",
    duration_ms: float | None = None,
    ok: bool | None = None,
    error: str = "",
    tool_call_id: str = "",
) -> dict[str, Any] | None:
    """Persist one tool invocation into ``chats/tool_call.json``.

    Session chat text stays in ``chats/sessions/``; all tool audits live in
    the shared ``tool_call.json`` (linked by optional ``session_id``).
    """
    from chat_log_core import _path_for, _read, _write
    name = (tool or "").strip()
    if not name:
        return None

    cid = (call_id or get_current_call_id() or "").strip()
    args = arguments if isinstance(arguments, dict) else {}
    entry: dict[str, Any] = {
        "at": _utc_iso(),
        "type": "tool_call",
        "source": (source or "server").strip() or "server",
        "tool": name,
        "arguments": _truncate_tool_payload(args, limit=2000),
    }
    if cid:
        entry["session_id"] = cid
    if tool_call_id:
        entry["tool_call_id"] = str(tool_call_id)[:120]
    if agent:
        entry["agent"] = str(agent).strip()
    if duration_ms is not None:
        try:
            entry["duration_ms"] = round(float(duration_ms), 1)
        except (TypeError, ValueError):
            pass
    if error:
        entry["ok"] = False
        entry["error"] = str(error)[:500]
    else:
        truncated = _truncate_tool_payload(result, limit=4000)
        entry["result"] = truncated
        if ok is None:
            if isinstance(truncated, dict) and "ok" in truncated:
                entry["ok"] = bool(truncated.get("ok"))
            else:
                entry["ok"] = True
        else:
            entry["ok"] = bool(ok)

    with _LOCK:
        store = _read_tool_call_store()
        store.setdefault("tool_calls", []).append(entry)
        _write_tool_call_store(store)

    return entry

def record_tool_calls_from_messages(
    messages: list[Any] | None,
    *,
    source: str = "langgraph",
    call_id: str | None = None,
    agent: str = "",
) -> int:
    """Scan LangGraph/AI messages for tool_calls + ToolMessage results and log them."""
    from chat_log_core import _path_for, _read, _write
    if not messages:
        return 0

    pending: dict[str, dict[str, Any]] = {}
    logged = 0

    for msg in messages:
        role = str(getattr(msg, "type", None) or getattr(msg, "role", None) or "").lower()
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            additional = getattr(msg, "additional_kwargs", None) or {}
            if isinstance(additional, dict):
                tool_calls = additional.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_id = str(tc.get("id") or tc.get("tool_call_id") or "")
                    name = str(tc.get("name") or tc.get("function", {}).get("name") or "")
                    raw_args = tc.get("args")
                    if raw_args is None and isinstance(tc.get("function"), dict):
                        raw_args = tc["function"].get("arguments")
                else:
                    tc_id = str(getattr(tc, "id", "") or "")
                    name = str(getattr(tc, "name", "") or "")
                    raw_args = getattr(tc, "args", None)
                args: dict[str, Any] = {}
                if isinstance(raw_args, dict):
                    args = raw_args
                elif isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                        if isinstance(parsed, dict):
                            args = parsed
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": raw_args}
                key = tc_id or f"{name}:{logged}:{len(pending)}"
                pending[key] = {
                    "tool": name,
                    "arguments": args,
                    "tool_call_id": tc_id,
                }

        is_tool = role in ("tool", "function") or (
            getattr(msg, "name", None) and role != "ai" and role != "assistant"
            and getattr(msg, "content", None) is not None
            and type(msg).__name__.lower().endswith("toolmessage")
        )
        if not is_tool and type(msg).__name__ == "ToolMessage":
            is_tool = True
        if not is_tool:
            continue

        tc_id = str(
            getattr(msg, "tool_call_id", None)
            or getattr(msg, "id", None)
            or ""
        )
        name = str(getattr(msg, "name", None) or getattr(msg, "tool", None) or "").strip()
        content = getattr(msg, "content", None)
        if isinstance(content, (dict, list)):
            result = content
        else:
            text = content if isinstance(content, str) else str(content or "")
            try:
                result = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                result = text

        meta = pending.pop(tc_id, None) if tc_id else None
        if meta is None and pending and name:
            # Match by tool name if ids are missing.
            for k, v in list(pending.items()):
                if v.get("tool") == name:
                    meta = pending.pop(k)
                    break
        if meta is None:
            meta = {"tool": name or "unknown", "arguments": {}, "tool_call_id": tc_id}

        record_tool_call(
            tool=str(meta.get("tool") or name or "unknown"),
            arguments=meta.get("arguments") if isinstance(meta.get("arguments"), dict) else {},
            result=result,
            source=source,
            call_id=call_id,
            agent=agent,
            tool_call_id=str(meta.get("tool_call_id") or tc_id or ""),
        )
        logged += 1

    # Tool calls that never got a ToolMessage (failed / aborted).
    for meta in pending.values():
        record_tool_call(
            tool=str(meta.get("tool") or "unknown"),
            arguments=meta.get("arguments") if isinstance(meta.get("arguments"), dict) else {},
            result=None,
            source=source,
            call_id=call_id,
            agent=agent,
            ok=False,
            error="no_tool_result",
            tool_call_id=str(meta.get("tool_call_id") or ""),
        )
        logged += 1

    return logged
