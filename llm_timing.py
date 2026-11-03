"""Per-LLM-call timing: where the LLM part of a reply's time goes.

Each chat-model HTTP request is stamped at four points (``perf_counter``):

* **T1** request sent      — httpx request hook, just before the bytes go out
* **T2** first byte        — httpx response hook, response headers received
* **T3** first token       — first streamed chunk carrying text or a tool call
* **T4** last token        — last streamed chunk carrying text or a tool call

T2−T1 is network + provider accepting the request; T3−T2 is the model reading
the prompt before its first token (usually the biggest part); T4−T3 is
streaming the rest of the output. T5 (first TTS audio byte) is
stamped by ``turn_metrics.py``, which pulls these calls for its turn via
``drain()`` and writes the per-turn summary.

Every call (including losing hedge copies) is appended to
``metrics/llm_calls.csv``.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CSV_PATH = Path(__file__).parent / "metrics" / "llm_calls.csv"
LLM_CALLS_CSV = Path(os.environ.get("LLM_CALLS_CSV", DEFAULT_CSV_PATH))

CSV_HEADER = [
    "session_id",
    "speculative",
    "node",
    "model",
    "copy",
    "status",
    "started_at",
    "t1_request_sent_ms",
    "t2_first_byte_ms",
    "t3_first_token_ms",
    "t4_last_token_ms",
    "wait_ms",
    "first_token_ms",
    "stream_ms",
    "total_ms",
    "output_chunks",
    "tool_calls",
]

_LOCK = threading.Lock()
_PENDING: dict[str, list["LLMCallTiming"]] = defaultdict(list)
_MAX_PENDING = 200


def streaming_enabled() -> bool:
    """Stream chat completions so first and last token can be told apart."""
    return os.getenv("LLM_STREAM", "1").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Scope:
    session_id: str = ""
    speculative: bool = False
    #: Called with each finished call (speculation counts a guess's calls).
    on_call: Any = None


#: Set by ``run_turn`` in the agent's worker thread; LangGraph copies it into nodes.
_SCOPE: ContextVar[Scope | None] = ContextVar("llm_timing_scope", default=None)
#: The call being made, inside the transport loop task (read by the httpx hooks).
_CALL: ContextVar["LLMCallTiming | None"] = ContextVar("llm_timing_call", default=None)


def set_scope(session_id: str, *, speculative: bool = False, on_call: Any = None) -> None:
    _SCOPE.set(Scope(session_id=session_id or "", speculative=speculative, on_call=on_call))


def current_scope() -> Scope:
    return _SCOPE.get() or Scope()


def current_node() -> str:
    """LangGraph node name (router, booking, ...) when called inside the graph."""
    try:
        from langgraph.config import get_config

        meta = get_config().get("metadata") or {}
        # Specialists are prebuilt ReAct subgraphs whose inner node is "agent";
        # the outer graph node (booking, cancelling, ...) leads the namespace.
        ns = str(meta.get("langgraph_checkpoint_ns") or "")
        outer = ns.split("|")[0].split(":")[0] if ns else ""
        return outer or str(meta.get("langgraph_node") or "")
    except Exception:
        return ""


@dataclass
class LLMCallTiming:
    session_id: str = ""
    speculative: bool = False
    node: str = ""
    model: str = ""
    copy: int = 1
    status: str = "pending"
    started_wall: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    t0: float = field(default_factory=time.perf_counter)
    t1: float | None = None
    t2: float | None = None
    t3: float | None = None
    t4: float | None = None
    chunks: int = 0
    tool_calls: int = 0
    on_call: Any = field(default=None, repr=False)

    def mark_request(self) -> None:
        if self.t1 is None:
            self.t1 = time.perf_counter()

    def mark_first_byte(self) -> None:
        if self.t2 is None:
            self.t2 = time.perf_counter()

    def mark_chunk(self, chunk: Any) -> None:
        """Stamp T3/T4 on chunks that carry generated text or tool-call pieces."""
        msg = getattr(chunk, "message", chunk)
        content = getattr(msg, "content", "")
        tool_bits = getattr(msg, "tool_call_chunks", None) or []
        if not content and not tool_bits:
            return
        now = time.perf_counter()
        if self.t3 is None:
            self.t3 = now
        self.t4 = now
        self.chunks += 1

    def mark_body(self, result: Any = None) -> None:
        """Non-streamed reply: the whole body arrives at once, so T3 = T4."""
        now = time.perf_counter()
        self.t3 = self.t3 or now
        self.t4 = now
        self.chunks = self.chunks or 1
        self._count_tools(result)

    def _count_tools(self, result: Any) -> None:
        try:
            msg = result.generations[0].message
            self.tool_calls = len(getattr(msg, "tool_calls", None) or [])
        except Exception:
            pass

    def finish(self, status: str, result: Any = None) -> None:
        self.status = status
        if result is not None:
            self._count_tools(result)
        _record(self)
        if self.on_call is not None:
            try:
                self.on_call(self)
            except Exception:
                pass

    # durations ------------------------------------------------------------
    @property
    def wait_ms(self) -> float | None:
        return _ms(self.t1, self.t2)

    @property
    def first_token_ms(self) -> float | None:
        return _ms(self.t2, self.t3)

    @property
    def stream_ms(self) -> float | None:
        return _ms(self.t3, self.t4)

    @property
    def total_ms(self) -> float | None:
        return _ms(self.t1, self.t4)


def _ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((b - a) * 1000, 1)


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.1f}"


# httpx hooks (installed on the shared client in llm_transport.py) ----------


async def on_request(request) -> None:  # noqa: ANN001 - httpx.Request
    call = _CALL.get()
    if call is not None:
        call.mark_request()


async def on_response(response) -> None:  # noqa: ANN001 - httpx.Response
    call = _CALL.get()
    if call is not None:
        call.mark_first_byte()


def begin(scope: Scope, *, node: str, model: str, copy: int) -> tuple[LLMCallTiming, Any]:
    call = LLMCallTiming(
        session_id=scope.session_id,
        speculative=scope.speculative,
        node=node,
        model=model,
        copy=copy,
        on_call=scope.on_call,
    )
    return call, _CALL.set(call)


def end(token: Any) -> None:
    _CALL.reset(token)


# storage --------------------------------------------------------------------


def _record(call: LLMCallTiming) -> None:
    row = {
        "session_id": call.session_id,
        "speculative": int(call.speculative),
        "node": call.node,
        "model": call.model,
        "copy": call.copy,
        "status": call.status,
        "started_at": call.started_wall.isoformat(timespec="milliseconds"),
        "t1_request_sent_ms": _fmt(_ms(call.t0, call.t1)),
        "t2_first_byte_ms": _fmt(_ms(call.t0, call.t2)),
        "t3_first_token_ms": _fmt(_ms(call.t0, call.t3)),
        "t4_last_token_ms": _fmt(_ms(call.t0, call.t4)),
        "wait_ms": _fmt(call.wait_ms),
        "first_token_ms": _fmt(call.first_token_ms),
        "stream_ms": _fmt(call.stream_ms),
        "total_ms": _fmt(call.total_ms),
        "output_chunks": call.chunks,
        "tool_calls": call.tool_calls,
    }
    with _LOCK:
        if call.status == "ok" and call.session_id:
            pending = _PENDING[call.session_id]
            pending.append(call)
            del pending[:-_MAX_PENDING]
        try:
            LLM_CALLS_CSV.parent.mkdir(parents=True, exist_ok=True)
            new = not LLM_CALLS_CSV.exists() or LLM_CALLS_CSV.stat().st_size == 0
            with LLM_CALLS_CSV.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_HEADER)
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError:
            pass


def drain(session_id: str) -> list[LLMCallTiming]:
    """Completed calls for a session since the last drain (oldest first)."""
    with _LOCK:
        return _PENDING.pop(session_id, [])


def summarize_turn(calls: list[LLMCallTiming], t0: float | None) -> dict[str, str]:
    """Per-turn T1–T4 relative to the turn start ``t0`` (perf_counter).

    One reply can take several LLM calls (router, specialist, tool rounds).
    T1/T2/T3 come from the first call, T4 from the last. A speculative guess
    that was discarded is ignored when the turn has its own calls.
    """
    real = [c for c in calls if not c.speculative]
    use = sorted(real or calls, key=lambda c: c.t1 or c.t0)
    empty = {k: "" for k in TURN_COLUMNS}
    if not use or t0 is None:
        return empty
    first, last = use[0], use[-1]
    llm_ms = sum((c.total_ms or 0.0) for c in use)
    span = _ms(first.t1, last.t4)
    return {
        "llm_calls": str(len(use)),
        "t1_llm_request_ms": _fmt(_ms(t0, first.t1)),
        "t2_llm_first_byte_ms": _fmt(_ms(t0, first.t2)),
        "t3_llm_first_token_ms": _fmt(_ms(t0, first.t3)),
        "t4_llm_last_token_ms": _fmt(_ms(t0, last.t4)),
        "llm_wait_ms": _fmt(sum((c.wait_ms or 0.0) for c in use)),
        "llm_first_token_ms": _fmt(sum((c.first_token_ms or 0.0) for c in use)),
        "llm_stream_ms": _fmt(sum((c.stream_ms or 0.0) for c in use)),
        "between_llm_calls_ms": _fmt(None if span is None else max(0.0, span - llm_ms)),
        "llm_nodes": "+".join(c.node or "?" for c in use),
    }


TURN_COLUMNS = [
    "llm_calls",
    "t1_llm_request_ms",
    "t2_llm_first_byte_ms",
    "t3_llm_first_token_ms",
    "t4_llm_last_token_ms",
    "llm_wait_ms",
    "llm_first_token_ms",
    "llm_stream_ms",
    "between_llm_calls_ms",
    "llm_nodes",
]
