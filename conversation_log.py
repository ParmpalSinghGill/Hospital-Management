"""Per-session conversation JSON logs in the product schema.

Layout under ``chats/``::

    chats/
      sessions/          # one JSON file per call/session
        sess-YYYYMMDD-HHMMSS_<session_id>.json
      tool_call.json     # all tool invocations (time, args, result)
      timeline.jsonl     # optional cross-session interrupt timeline

Implementation is split across:
  chat_log_core.py, chat_log_turns.py, chat_log_tools.py,
  chat_log_events.py, chat_log_query.py
"""

from __future__ import annotations

from chat_log_core import (  # noqa: F401
    CHATS_DIR,
    SESSIONS_DIR,
    TIMELINE_PATH,
    TOOL_CALL_PATH,
    _CURRENT_CALL_ID,
    _LOCK,
    _MIGRATED,
    _OPEN,
    _ROOT,
    _ensure_layout,
    _estimate_tokens,
    _iso_from_ms,
    _load_tool_call_file,
    _path_for,
    _read,
    _read_tool_call_store,
    _safe_id,
    _save_tool_call_file,
    _utc_iso,
    _write,
    _write_tool_call_store,
    get_current_call_id,
    set_current_call_id,
)
from chat_log_turns import (  # noqa: F401
    _PENDING_TURN_META,
    _channel_label,
    _empty_interaction,
    _interactions,
    _refresh_interaction,
    _set_text,
    _set_timestamp,
    append_or_update_turn,
    end_call,
    link_call_patient,
    note_turn_meta,
    record_server_turn,
    start_call,
)
from chat_log_tools import (  # noqa: F401
    _truncate_tool_payload,
    list_tool_calls,
    record_tool_call,
    record_tool_calls_from_messages,
)
from chat_log_events import (  # noqa: F401
    apply_client_event,
    record_interrupt,
    record_timeline_event,
)
from chat_log_query import (  # noqa: F401
    _chat_matches_patient,
    _chat_preview_turns,
    _infer_legacy_agent,
    _iter_chat_files,
    _normalize_digits,
    delete_call,
    get_call_detail,
    list_calls_for_patient,
    list_recent_calls,
    record_booking_chat,
    session_response_timings,
    summarize_response_timings,
)
