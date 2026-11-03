"""Low-latency LLM transport: shared HTTP/2 client + hedged (raced) requests.

Every chat-model call is executed on one long-lived background event loop that
owns a single ``httpx.AsyncClient``. That gives us:

- **Keep-alive** — TCP + TLS handshakes happen once per provider host, not per call.
- **HTTP/2** (``LLM_HTTP2=1``) — parallel requests multiplex over one connection,
  and cancelling a losing stream does not tear the connection down.
- **Hedging** (``LLM_HEDGE=N``) — fire N identical requests, keep the first
  successful response, cancel the rest. ``LLM_HEDGE_DELAY_MS`` staggers the
  extra copies (classic hedged requests: only pay for a duplicate when the
  first one is slow). Hedging multiplies token cost by up to N.

The LangGraph agents call ``model.invoke`` from worker threads; sync calls are
bridged onto the background loop, so the async client is only ever used from
the loop it was created on.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Optional

import httpx

import llm_timing

log = logging.getLogger(__name__)

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()
_CLIENTS: dict[bool, httpx.AsyncClient] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def http2_enabled() -> bool:
    return os.getenv("LLM_HTTP2", "1").strip().lower() not in ("0", "false", "no", "off")


def hedge_count() -> int:
    return max(1, min(_env_int("LLM_HEDGE", 1), 5))


def hedge_delay_ms() -> int:
    return max(0, _env_int("LLM_HEDGE_DELAY_MS", 0))


def background_loop() -> asyncio.AbstractEventLoop:
    """The one event loop that owns all LLM HTTP connections."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="llm-transport", daemon=True).start()
            _LOOP = loop
        return _LOOP


def async_http_client(http2: Optional[bool] = None) -> httpx.AsyncClient:
    """Shared pooled client (one per protocol). Must only be awaited on ``background_loop()``."""
    h2 = http2_enabled() if http2 is None else http2
    with _LOOP_LOCK:
        client = _CLIENTS.get(h2)
        if client is None:
            client = httpx.AsyncClient(
                http2=h2,
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=120),
                # T1 (request sent) / T2 (first byte) per call — see llm_timing.py
                event_hooks={"request": [llm_timing.on_request], "response": [llm_timing.on_response]},
            )
            _CLIENTS[h2] = client
        return client


async def _race(make_call, n: int, delay_ms: int, label: str):
    """Run ``make_call()`` n times, return the first success, cancel the rest."""
    if n <= 1:
        return await make_call()

    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []
    pending: set[asyncio.Task] = set()
    errors: list[BaseException] = []

    def launch() -> None:
        task = asyncio.ensure_future(make_call())
        tasks.append(task)
        pending.add(task)

    for _ in range(1 if delay_ms else n):
        launch()
    try:
        while pending or len(tasks) < n:
            if not pending:
                launch()  # every copy so far failed — try the next one now
                continue
            timeout = delay_ms / 1000 if len(tasks) < n else None
            done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                launch()  # stagger elapsed with no answer yet — send another copy
                continue
            for task in done:
                pending.discard(task)
                if task.exception() is None:
                    log.debug(
                        "hedge %s: copy %d/%d won in %.0f ms",
                        label, tasks.index(task) + 1, len(tasks), (time.perf_counter() - t0) * 1000,
                    )
                    return task.result()
                errors.append(task.exception())
        raise errors[-1]
    finally:
        for task in pending:
            task.cancel()


class HedgedChatMixin:
    """Mix into a LangChain chat model whose async client uses ``async_http_client()``.

    Concrete classes declare ``hedge_count`` / ``hedge_delay_ms`` pydantic fields.
    """

    def _hedged(self, messages, stop, kwargs, scope=None, node: str = ""):
        from langchain_core.language_models.chat_models import agenerate_from_stream

        parent_agenerate = super()._agenerate  # type: ignore[misc]
        parent_astream = super()._astream  # type: ignore[misc]
        model = str(getattr(self, "model_name", "") or getattr(self, "model", ""))
        scope = scope or llm_timing.Scope()
        stream = llm_timing.streaming_enabled()
        copies = 0

        async def timed_call():
            # Runs as its own task: the timing ContextVar is private to this copy.
            nonlocal copies
            copies += 1
            call, token = llm_timing.begin(scope, node=node, model=model, copy=copies)
            try:
                if stream:
                    async def chunks():
                        async for chunk in parent_astream(messages, stop=stop, run_manager=None, **kwargs):
                            call.mark_chunk(chunk)
                            yield chunk

                    result = await agenerate_from_stream(chunks())
                else:
                    result = await parent_agenerate(messages, stop=stop, run_manager=None, **kwargs)
                    call.mark_body(result)
            except asyncio.CancelledError:
                call.finish("cancelled")
                raise
            except BaseException:
                call.finish("error")
                raise
            finally:
                llm_timing.end(token)
            call.finish("ok", result)
            return result

        return _race(
            timed_call,
            self.hedge_count,  # type: ignore[attr-defined]
            self.hedge_delay_ms,  # type: ignore[attr-defined]
            model,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        # Scope and node live in the caller's context; the loop thread can't see them.
        coro = self._hedged(messages, stop, kwargs, llm_timing.current_scope(), llm_timing.current_node())
        fut = asyncio.run_coroutine_threadsafe(coro, background_loop())
        return fut.result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        coro = self._hedged(messages, stop, kwargs, llm_timing.current_scope(), llm_timing.current_node())
        fut = asyncio.run_coroutine_threadsafe(coro, background_loop())
        return await asyncio.wrap_future(fut)
