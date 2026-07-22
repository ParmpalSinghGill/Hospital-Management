"""Per-LLM-call request/response dumps (admin-gated).

When Admin → Settings → "Save LLM messages" is on, a LangChain callback
records **each** model call:

    chats/messages/<session_id>/Message{n}_passed.txt   # exact messages to LLM
    chats/messages/<session_id>/Message{n}_response.txt # exact LLM response

Use ``llm_dump_callbacks(node)`` in agent ``invoke(..., config=...)``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

_ROOT = Path(__file__).resolve().parent
MESSAGES_ROOT = _ROOT / "chats" / "messages"

_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {}
_SAFE_SESSION = re.compile(r"[^A-Za-z0-9._-]+")


def save_llm_messages_enabled() -> bool:
    try:
        from service_settings import load_settings

        return bool(load_settings().get("save_llm_messages", False))
    except Exception:
        return False


SESSION_META_NAME = "session_meta.json"


def vad_runtime_meta(*, start_secs: float | None = None, stop_secs: float | None = None) -> dict[str, Any]:
    """Current VAD timing used for voice turns (Admin ``vad_stop_secs`` + fixed start)."""
    try:
        from service_settings import clamp_vad_stop_secs, load_settings

        stop = clamp_vad_stop_secs(
            stop_secs if stop_secs is not None else load_settings().get("vad_stop_secs", 0.2)
        )
    except Exception:
        try:
            stop = float(stop_secs) if stop_secs is not None else 0.2
        except (TypeError, ValueError):
            stop = 0.2
        stop = max(0.1, min(3.0, round(stop, 2)))
    try:
        start = float(start_secs) if start_secs is not None else 0.2
    except (TypeError, ValueError):
        start = 0.2
    start = max(0.05, min(3.0, round(start, 2)))
    return {
        "start_secs": start,
        "stop_secs": stop,
    }


def write_session_runtime_meta(
    session_id: str,
    *,
    start_secs: float | None = None,
    stop_secs: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Persist per-call VAD timing under the LLM messages session folder."""
    if not save_llm_messages_enabled():
        return None
    try:
        sid = _resolve_session_id(session_id)
        folder = _session_dir(sid)
        meta = {
            "session_id": sid,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            **vad_runtime_meta(start_secs=start_secs, stop_secs=stop_secs),
        }
        if extra:
            meta.update({k: v for k, v in extra.items() if v is not None})
        (folder / SESSION_META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return meta
    except Exception:
        logging.exception("Failed to write session runtime meta")
        return None


def read_session_runtime_meta(session_id: str) -> dict[str, Any]:
    folder = _session_dir(session_id, create=False)
    path = folder / SESSION_META_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record_client_turn_timing(session_id: str, patch: dict[str, Any] | None) -> None:
    """Append first-text / first-audio timing into the LLM messages session folder.

    Also merges onto the latest Message*_meta.json when present so Admin dumps
    keep the utterance + latency next to each LLM call.
    """
    if not save_llm_messages_enabled() or not patch:
        return
    interesting = any(
        patch.get(k) is not None
        for k in (
            "bot_text_first_shown_at",
            "bot_voice_first_heard_at",
            "first_text",
            "first_speech",
            "first_text_latency_ms",
            "first_audio_latency_ms",
        )
    )
    if not interesting:
        return
    try:
        sid = _resolve_session_id(session_id)
        folder = _session_dir(sid)
        entry: dict[str, Any] = {
            "session_id": sid,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }
        for key in (
            "bot_text_first_shown_at",
            "bot_voice_first_heard_at",
            "first_text",
            "first_speech",
            "first_text_latency_ms",
            "first_audio_latency_ms",
        ):
            if patch.get(key) is not None:
                entry[key] = patch[key]
        with _LOCK:
            path = folder / "turn_timings.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            metas = sorted(folder.glob("Message*_meta.json"), key=lambda p: p.name)
            if metas:
                latest = metas[-1]
                try:
                    meta = json.loads(latest.read_text(encoding="utf-8"))
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
                client = meta.setdefault("client_timings", {})
                if not isinstance(client, dict):
                    client = {}
                    meta["client_timings"] = client
                if entry.get("first_text") is not None:
                    client["first_text"] = entry["first_text"]
                if entry.get("first_speech") is not None:
                    client["first_speech"] = entry["first_speech"]
                if entry.get("first_text_latency_ms") is not None:
                    client["first_text_latency_ms"] = entry["first_text_latency_ms"]
                if entry.get("first_audio_latency_ms") is not None:
                    client["first_audio_latency_ms"] = entry["first_audio_latency_ms"]
                if entry.get("bot_text_first_shown_at"):
                    client["first_text_at"] = entry["bot_text_first_shown_at"]
                if entry.get("bot_voice_first_heard_at"):
                    client["first_audio_at"] = entry["bot_voice_first_heard_at"]
                latest.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except Exception:
        logging.exception("Failed to record client turn timing")


def delete_session_messages(session_id: str) -> bool:
    """Delete one session's LLM dump folder. Returns True if it existed."""
    import shutil

    folder = _session_dir(session_id, create=False)
    with _LOCK:
        _COUNTERS.pop(session_id, None)
        safe = folder.name
        _COUNTERS.pop(safe, None)
    if not folder.exists() or not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True


def delete_all_llm_messages() -> int:
    """Delete every LLM dump session folder. Returns number of folders removed."""
    import shutil

    if not MESSAGES_ROOT.exists():
        return 0
    removed = 0
    with _LOCK:
        _COUNTERS.clear()
    for folder in list(MESSAGES_ROOT.iterdir()):
        if folder.is_dir():
            shutil.rmtree(folder)
            removed += 1
    return removed


def _session_dir(session_id: str, *, create: bool = True) -> Path:
    raw = (session_id or "unknown").strip() or "unknown"
    safe = _SAFE_SESSION.sub("_", raw)[:120]
    path = MESSAGES_ROOT / safe
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_session_id(explicit: str | None = None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    try:
        from conversation_log import get_current_call_id

        cid = get_current_call_id()
        if cid:
            return str(cid)
    except Exception:
        pass
    return "unknown"


def _next_n(session_id: str) -> int:
    folder = _session_dir(session_id)
    with _LOCK:
        if session_id not in _COUNTERS:
            existing = 0
            for p in folder.glob("Message*_passed.txt"):
                m = re.match(r"Message(\d+)_passed\.txt$", p.name)
                if m:
                    existing = max(existing, int(m.group(1)))
            _COUNTERS[session_id] = existing
        _COUNTERS[session_id] += 1
        return _COUNTERS[session_id]


def message_to_dict(msg: Any) -> dict[str, Any]:
    if isinstance(msg, tuple) and msg:
        out: dict[str, Any] = {"role": str(msg[0] or "unknown")}
        out["content"] = msg[1] if len(msg) > 1 else ""
        return out
    if isinstance(msg, dict):
        out = {
            "role": str(msg.get("role") or msg.get("type") or "unknown"),
            "content": msg.get("content", ""),
        }
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            out["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            out["name"] = msg["name"]
        return out

    role = getattr(msg, "type", None) or getattr(msg, "role", None) or type(msg).__name__
    out = {"role": str(role), "content": getattr(msg, "content", "")}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = tool_calls
    additional = getattr(msg, "additional_kwargs", None)
    if isinstance(additional, dict) and additional.get("tool_calls"):
        out.setdefault("tool_calls", additional["tool_calls"])
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    name = getattr(msg, "name", None)
    if name:
        out["name"] = name
    msg_id = getattr(msg, "id", None)
    if msg_id:
        out["id"] = str(msg_id)
    return out


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(value)


def format_messages_block(messages: list[Any], *, title: str, node: str, n: int) -> str:
    lines = [
        f"llm_call: {n}",
        f"node: {node}",
        f"kind: {title}",
        f"message_count: {len(messages)}",
        f"saved_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')}",
        "",
        "===== EXACT PAYLOAD (JSON) =====",
        json.dumps([message_to_dict(m) for m in messages], ensure_ascii=False, indent=2, default=str),
        "",
        "===== READABLE =====",
    ]
    for i, msg in enumerate(messages, start=1):
        data = message_to_dict(msg)
        lines.append(f"--- message {i}/{len(messages)} role={data.get('role')} ---")
        lines.append(_stringify(data.get("content")))
        if data.get("tool_calls"):
            lines.append("-- tool_calls --")
            lines.append(_stringify(data.get("tool_calls")))
        if data.get("tool_call_id"):
            lines.append(f"tool_call_id: {data['tool_call_id']}")
        if data.get("name"):
            lines.append(f"name: {data['name']}")
        lines.append("")
    return "\n".join(lines)


def format_response_block(response: Any, *, node: str, n: int) -> str:
    """Serialize LangChain LLMResult / chat generations."""
    generations: list[Any] = []
    llm_output: Any = None
    if hasattr(response, "generations"):
        generations = list(response.generations or [])
        llm_output = getattr(response, "llm_output", None)
    elif isinstance(response, dict):
        generations = list(response.get("generations") or [])
        llm_output = response.get("llm_output")

    flat: list[dict[str, Any]] = []
    readable: list[str] = []
    for gi, gens in enumerate(generations):
        for g in gens if isinstance(gens, list) else [gens]:
            msg = getattr(g, "message", None)
            text = getattr(g, "text", None)
            if msg is not None:
                data = message_to_dict(msg)
                flat.append(data)
                readable.append(f"--- generation[{gi}] role={data.get('role')} ---")
                readable.append(_stringify(data.get("content")))
                if data.get("tool_calls"):
                    readable.append("-- tool_calls --")
                    readable.append(_stringify(data.get("tool_calls")))
                readable.append("")
            elif text is not None:
                flat.append({"role": "assistant", "content": text})
                readable.append(f"--- generation[{gi}] text ---")
                readable.append(str(text))
                readable.append("")
            else:
                flat.append({"raw": str(g)})
                readable.append(str(g))

    lines = [
        f"llm_call: {n}",
        f"node: {node}",
        f"kind: RESPONSE",
        f"generation_count: {len(flat)}",
        f"saved_at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')}",
        "",
        "===== EXACT RESPONSE (JSON) =====",
        json.dumps(
            {"generations": flat, "llm_output": llm_output},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
        "===== READABLE =====",
        *readable,
    ]
    return "\n".join(lines)


def save_llm_pair(
    *,
    passed_messages: list[Any],
    response: Any,
    node: str = "",
    session_id: str | None = None,
) -> int | None:
    """Write Message{n}_passed.txt + Message{n}_response.txt. Returns n or None."""
    if not save_llm_messages_enabled():
        return None
    try:
        sid = _resolve_session_id(session_id)
        n = _next_n(sid)
        folder = _session_dir(sid)
        (folder / f"Message{n}_passed.txt").write_text(
            format_messages_block(passed_messages, title="PASSED_TO_LLM", node=node, n=n),
            encoding="utf-8",
        )
        (folder / f"Message{n}_response.txt").write_text(
            format_response_block(response, node=node, n=n),
            encoding="utf-8",
        )
        meta = {
            "n": n,
            "node": node,
            "session_id": sid,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "passed_file": f"Message{n}_passed.txt",
            "response_file": f"Message{n}_response.txt",
            "passed_message_count": len(passed_messages),
            **vad_runtime_meta(),
        }
        # Prefer per-call session meta written at voice start (exact pipeline values).
        session_meta = read_session_runtime_meta(sid)
        for key in ("start_secs", "stop_secs", "confidence", "min_volume", "pipeline_mode"):
            if key in session_meta and session_meta[key] is not None:
                meta[key] = session_meta[key]
        (folder / f"Message{n}_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        logging.info("LLM dump saved session=%s n=%s node=%s", sid, n, node)
        return n
    except Exception:
        logging.exception("Failed to save LLM pair")
        return None


def llm_dump_callbacks(node: str = "") -> list[Any]:
    """Return LangChain callbacks that dump each chat-model call (or [])."""
    if not save_llm_messages_enabled():
        return []
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:
        return []

    class _LlmDumpCallback(BaseCallbackHandler):
        """Capture exact chat messages in → LLM generations out, per call."""

        def __init__(self, agent_node: str):
            super().__init__()
            self.agent_node = agent_node
            self._pending: dict[str, list[Any]] = {}

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            # messages is a batch: List[List[BaseMessage]]
            flat: list[Any] = []
            for batch in messages or []:
                flat.extend(list(batch or []))
            self._pending[str(run_id)] = flat

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            # Non-chat LLMs: store prompts as synthetic user messages.
            if str(run_id) not in self._pending:
                self._pending[str(run_id)] = [("user", p) for p in (prompts or [])]

        def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
            key = str(run_id)
            passed = self._pending.pop(key, None)
            if passed is None:
                passed = []
            save_llm_pair(
                passed_messages=passed,
                response=response,
                node=self.agent_node,
            )

        def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            self._pending.pop(str(run_id), None)

    return [_LlmDumpCallback(node)]


def agent_invoke_config(node: str, *, recursion_limit: int) -> dict[str, Any]:
    """Config dict for agent.invoke with optional LLM dump callbacks."""
    cfg: dict[str, Any] = {"recursion_limit": recursion_limit}
    cbs = llm_dump_callbacks(node)
    if cbs:
        cfg["callbacks"] = cbs
    return cfg


# ----- Admin listing helpers -----


def list_llm_message_sessions(*, oldest_first: bool = True) -> list[dict[str, Any]]:
    """List dump sessions. Default: oldest at top."""
    if not MESSAGES_ROOT.exists():
        return []
    try:
        from conversation_log import session_response_timings
    except Exception:
        session_response_timings = None  # type: ignore[assignment]

    rows: list[dict[str, Any]] = []
    for folder in MESSAGES_ROOT.iterdir():
        if not folder.is_dir():
            continue
        pairs = list_session_pairs(folder.name)
        if not pairs:
            continue
        mtimes = [p.stat().st_mtime for p in folder.glob("Message*_passed.txt")]
        session_meta = read_session_runtime_meta(folder.name)
        # Fall back to latest pair meta for older dumps.
        latest = pairs[-1] if pairs else {}
        avg_text = None
        avg_speech = None
        text_n = 0
        speech_n = 0
        # Prefer chats/sessions client timings; fall back to Message meta / turn_timings.
        chat_timings = session_response_timings(folder.name) if session_response_timings else None
        if chat_timings:
            avg_text = chat_timings.get("avg_first_text_ms")
            avg_speech = chat_timings.get("avg_first_speech_ms")
            text_n = int((chat_timings.get("first_text") or {}).get("count") or 0)
            speech_n = int((chat_timings.get("first_speech") or {}).get("count") or 0)
        if avg_text is None and avg_speech is None:
            text_samples: list[float] = []
            audio_samples: list[float] = []
            for meta in pairs:
                client = meta.get("client_timings") if isinstance(meta, dict) else None
                if not isinstance(client, dict):
                    continue
                try:
                    if client.get("first_text_latency_ms") is not None:
                        text_samples.append(float(client["first_text_latency_ms"]))
                except (TypeError, ValueError):
                    pass
                try:
                    if client.get("first_audio_latency_ms") is not None:
                        audio_samples.append(float(client["first_audio_latency_ms"]))
                except (TypeError, ValueError):
                    pass
            timing_log = folder / "turn_timings.jsonl"
            if timing_log.exists():
                try:
                    for line in timing_log.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if not isinstance(entry, dict):
                            continue
                        try:
                            if entry.get("first_text_latency_ms") is not None:
                                text_samples.append(float(entry["first_text_latency_ms"]))
                        except (TypeError, ValueError):
                            pass
                        try:
                            if entry.get("first_audio_latency_ms") is not None:
                                audio_samples.append(float(entry["first_audio_latency_ms"]))
                        except (TypeError, ValueError):
                            pass
                except Exception:
                    pass
            if text_samples:
                avg_text = round(sum(text_samples) / len(text_samples), 1)
                text_n = len(text_samples)
            if audio_samples:
                avg_speech = round(sum(audio_samples) / len(audio_samples), 1)
                speech_n = len(audio_samples)
        rows.append(
            {
                "session_id": folder.name,
                "pair_count": len(pairs),
                "first_at": pairs[0].get("at") if pairs else None,
                "last_at": pairs[-1].get("at") if pairs else None,
                "mtime": max(mtimes) if mtimes else folder.stat().st_mtime,
                "start_secs": session_meta.get("start_secs", latest.get("start_secs")),
                "stop_secs": session_meta.get("stop_secs", latest.get("stop_secs")),
                "avg_first_text_ms": avg_text,
                "avg_first_speech_ms": avg_speech,
                "first_text_samples": text_n,
                "first_speech_samples": speech_n,
            }
        )
    rows.sort(key=lambda r: r.get("mtime") or 0, reverse=not oldest_first)
    return rows


def list_session_pairs(session_id: str) -> list[dict[str, Any]]:
    folder = _session_dir(session_id, create=False)
    if not folder.exists():
        return []
    # Prefer reading meta; fall back to passed files.
    metas = sorted(folder.glob("Message*_meta.json"), key=lambda p: p.name)
    if metas:
        out = []
        for meta_path in metas:
            try:
                out.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return out
    pairs = []
    for passed in sorted(folder.glob("Message*_passed.txt"), key=lambda p: p.name):
        m = re.match(r"Message(\d+)_passed\.txt$", passed.name)
        if not m:
            continue
        n = int(m.group(1))
        pairs.append(
            {
                "n": n,
                "session_id": session_id,
                "passed_file": passed.name,
                "response_file": f"Message{n}_response.txt",
                "at": datetime.fromtimestamp(passed.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        )
    return pairs


def read_session_pair(session_id: str, n: int) -> dict[str, Any] | None:
    folder = _session_dir(session_id, create=False)
    if not folder.exists():
        return None
    passed_path = folder / f"Message{n}_passed.txt"
    response_path = folder / f"Message{n}_response.txt"
    meta_path = folder / f"Message{n}_meta.json"
    if not passed_path.exists() and not response_path.exists():
        return None
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    session_meta = read_session_runtime_meta(session_id)
    # Prefer call-start session meta (exact pipeline values) over pair snapshot.
    for key in ("start_secs", "stop_secs", "confidence", "min_volume", "pipeline_mode"):
        if session_meta.get(key) is not None:
            meta[key] = session_meta[key]
        elif key not in meta or meta.get(key) is None:
            continue
    if meta.get("start_secs") is None or meta.get("stop_secs") is None:
        defaults = vad_runtime_meta()
        meta.setdefault("start_secs", defaults["start_secs"])
        meta.setdefault("stop_secs", defaults["stop_secs"])
    return {
        "session_id": session_id,
        "n": n,
        "node": meta.get("node") or "",
        "at": meta.get("at") or "",
        "start_secs": meta.get("start_secs"),
        "stop_secs": meta.get("stop_secs"),
        "passed": passed_path.read_text(encoding="utf-8") if passed_path.exists() else "",
        "response": response_path.read_text(encoding="utf-8") if response_path.exists() else "",
        "meta": meta,
        "session_meta": session_meta,
    }
