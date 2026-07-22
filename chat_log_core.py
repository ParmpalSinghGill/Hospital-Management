"""Chat log paths, locks, and low-level JSON I/O."""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
CHATS_DIR = Path(os.environ.get("CHAT_LOGS_DIR", _ROOT / "chats"))
SESSIONS_DIR = CHATS_DIR / "sessions"
TOOL_CALL_PATH = CHATS_DIR / "tool_call.json"
TIMELINE_PATH = CHATS_DIR / "timeline.jsonl"
_LOCK = threading.RLock()
_OPEN: dict[str, Path] = {}
_CURRENT_CALL_ID = threading.local()
_MIGRATED = False


def set_current_call_id(call_id: str | None) -> None:
    """Bind the active voice/CLI call id so tools can tag the open chat log."""
    _CURRENT_CALL_ID.value = (call_id or "").strip()

def get_current_call_id() -> str:
    return str(getattr(_CURRENT_CALL_ID, "value", "") or "").strip()

def _utc_iso(dt: datetime | None = None) -> str:
    """UTC ISO-8601 with millisecond precision and trailing Z."""
    d = dt or datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone(timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"

def _iso_from_ms(ms: float | int | None) -> str | None:
    if ms is None:
        return None
    try:
        return _utc_iso(datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc))
    except (TypeError, ValueError, OSError):
        return None

def _safe_id(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in (value or "").strip())
    return cleaned[:80] or "unknown"

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words + punctuation); good enough for logs."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    parts = re.findall(r"\w+|[^\w\s]", cleaned, flags=re.UNICODE)
    return max(1, len(parts))

def _empty_tool_call_store() -> dict[str, Any]:
    return {
        "updated_at": None,
        "tool_calls": [],
    }

def _load_tool_call_file() -> dict[str, Any]:
    """Read ``tool_call.json`` without running layout migration (no recursion)."""
    if not TOOL_CALL_PATH.exists():
        return _empty_tool_call_store()
    try:
        data = json.loads(TOOL_CALL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_tool_call_store()
    if isinstance(data, list):
        return {"updated_at": _utc_iso(), "tool_calls": data}
    if not isinstance(data, dict):
        return _empty_tool_call_store()
    if not isinstance(data.get("tool_calls"), list):
        data["tool_calls"] = []
    return data

def _save_tool_call_file(data: dict[str, Any]) -> None:
    """Write ``tool_call.json`` without running layout migration."""
    calls = data.get("tool_calls")
    if not isinstance(calls, list):
        calls = []
        data["tool_calls"] = calls
    if len(calls) > 5000:
        data["tool_calls"] = calls[-5000:]
    data["updated_at"] = _utc_iso()
    data["count"] = len(data["tool_calls"])
    TOOL_CALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOOL_CALL_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(TOOL_CALL_PATH)

def _ensure_layout() -> None:
    """Create chats/sessions/ and migrate legacy flat files once per process."""
    global _MIGRATED
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if _MIGRATED:
        return
    with _LOCK:
        if _MIGRATED:
            return
        try:
            # Move legacy session JSONs from chats/ root → chats/sessions/
            for path in list(CHATS_DIR.glob("sess-*.json")) + list(CHATS_DIR.glob("call-*.json")):
                dest = SESSIONS_DIR / path.name
                if dest.exists():
                    continue
                try:
                    path.replace(dest)
                except OSError:
                    pass

            # Merge legacy tool_calls.jsonl → tool_call.json
            legacy_jsonl = CHATS_DIR / "tool_calls.jsonl"
            if legacy_jsonl.is_file():
                existing = _load_tool_call_file()
                seen = {
                    json.dumps(row, sort_keys=True, default=str)
                    for row in existing.get("tool_calls") or []
                    if isinstance(row, dict)
                }
                added = 0
                try:
                    for line in legacy_jsonl.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        key = json.dumps(row, sort_keys=True, default=str)
                        if key in seen:
                            continue
                        existing.setdefault("tool_calls", []).append(row)
                        seen.add(key)
                        added += 1
                except OSError:
                    pass
                if added or not TOOL_CALL_PATH.exists():
                    _save_tool_call_file(existing)
                try:
                    legacy_jsonl.rename(CHATS_DIR / "tool_calls.jsonl.bak")
                except OSError:
                    pass
            elif not TOOL_CALL_PATH.exists():
                _save_tool_call_file(_empty_tool_call_store())

            # One-time: lift any per-session ``tool_calls`` into the shared store.
            lift_marker = CHATS_DIR / ".tool_calls_lifted"
            if not lift_marker.exists():
                store = _load_tool_call_file()
                seen = {
                    json.dumps(row, sort_keys=True, default=str)
                    for row in store.get("tool_calls") or []
                    if isinstance(row, dict)
                }
                lifted = 0
                for path in list(SESSIONS_DIR.glob("sess-*.json")) + list(
                    SESSIONS_DIR.glob("call-*.json")
                ):
                    data = _read(path)
                    calls = data.get("tool_calls")
                    if not isinstance(calls, list) or not calls:
                        continue
                    sid = str(data.get("session_id") or data.get("call_id") or "").strip()
                    for row in calls:
                        if not isinstance(row, dict):
                            continue
                        entry = dict(row)
                        if sid and "session_id" not in entry:
                            entry["session_id"] = sid
                        key = json.dumps(entry, sort_keys=True, default=str)
                        if key in seen:
                            continue
                        store.setdefault("tool_calls", []).append(entry)
                        seen.add(key)
                        lifted += 1
                    data.pop("tool_calls", None)
                    data["updated_at"] = _utc_iso()
                    _write(path, data)
                if lifted:
                    _save_tool_call_file(store)
                try:
                    lift_marker.write_text("1\n", encoding="utf-8")
                except OSError:
                    pass
        finally:
            _MIGRATED = True

def _path_for(session_id: str) -> Path:
    _ensure_layout()
    if session_id in _OPEN:
        return _OPEN[session_id]
    sid = _safe_id(session_id)
    matches = sorted(SESSIONS_DIR.glob(f"*_{sid}.json")) + sorted(
        SESSIONS_DIR.glob(f"sess_*_{sid}.json")
    )
    # Also resolve legacy paths still under chats/ root during migration races.
    if not matches:
        matches = sorted(CHATS_DIR.glob(f"*_{sid}.json")) + sorted(
            CHATS_DIR.glob(f"sess_*_{sid}.json")
        )
    if matches:
        path = matches[-1]
        # Prefer sessions/ location going forward.
        if path.parent != SESSIONS_DIR:
            new_path = SESSIONS_DIR / path.name
            if not new_path.exists():
                try:
                    path.replace(new_path)
                    path = new_path
                except OSError:
                    pass
            else:
                path = new_path
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = SESSIONS_DIR / f"sess-{stamp}_{sid}.json"
    _OPEN[session_id] = path
    return path

def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def _read_tool_call_store() -> dict[str, Any]:
    _ensure_layout()
    return _load_tool_call_file()

def _write_tool_call_store(data: dict[str, Any]) -> None:
    _ensure_layout()
    _save_tool_call_file(data)
