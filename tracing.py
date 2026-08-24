"""Optional Langfuse tracing for LangGraph turns.

Off unless ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are set (and
``LANGFUSE_TRACING_ENABLED`` is not ``0``). When on, every ``run_turn`` call is
traced end-to-end: router → specialist agent → LLM calls → tool calls, grouped
into one Langfuse *session* per call/thread.

HIPAA posture: traces carry conversation text, so the default target is the
self-hosted docker instance (``LANGFUSE_BASE_URL``). While ``HIPAA_MODE=1`` we
also run payloads through :mod:`phi_redact` before anything leaves the process
(``LANGFUSE_MASK_PHI=0`` turns that off for synthetic data). Coverage matches the
log redactor: phones / e-mails are masked in any text, name fields (tool args &
results) are masked, but a name typed inside a free-text prompt or reply is not
— hence self-hosted is the default, not a cloud project.
Any SDK failure degrades to "no tracing" — it must never break a patient call.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, Iterable

import security_config as sc

_LOCK = threading.Lock()
_HANDLER: Any = None
_CLIENT: Any = None
_INIT_FAILED = False
_log = logging.getLogger(__name__)

DEFAULT_TRACE_NAME = "hospital_turn"


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def langfuse_configured() -> bool:
    return bool((os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip() and (os.getenv("LANGFUSE_SECRET_KEY") or "").strip())


def langfuse_enabled() -> bool:
    """Keys present and tracing not explicitly paused."""
    return langfuse_configured() and _env_flag("LANGFUSE_TRACING_ENABLED", True)


def mask_phi_enabled() -> bool:
    """Mask identifiers in traces while HIPAA mode is on (default), unless disabled."""
    return sc.hipaa_mode() and _env_flag("LANGFUSE_MASK_PHI", True)


def mask_for_trace(data: Any) -> Any:
    """Langfuse ``mask`` hook: irreversible PHI minimisation of trace payloads."""
    try:
        from phi_redact import redact_obj

        return redact_obj(data)
    except Exception:  # noqa: BLE001 — never leak raw data if the redactor fails
        return "[masked]"


def base_url() -> str:
    return (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").strip()


def get_handler() -> Any | None:
    """Process-wide Langfuse LangChain callback handler (or None when disabled)."""
    global _HANDLER, _CLIENT, _INIT_FAILED
    if not langfuse_enabled() or _INIT_FAILED:
        return None
    if _HANDLER is not None:
        return _HANDLER
    with _LOCK:
        if _HANDLER is not None:
            return _HANDLER
        try:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            kwargs: dict[str, Any] = {}
            if mask_phi_enabled():
                kwargs["mask"] = mask_for_trace
            env_name = (os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or "").strip()
            if env_name:
                kwargs["environment"] = env_name
            # Constructor registers the singleton client the handler uses; base URL /
            # keys come from LANGFUSE_* env vars.
            _CLIENT = Langfuse(**kwargs)
            _HANDLER = CallbackHandler()
            atexit.register(flush)
            _log.info(
                "Langfuse tracing on (%s, mask_phi=%s)", base_url() or "cloud", mask_phi_enabled()
            )
        except Exception as exc:  # noqa: BLE001
            _INIT_FAILED = True
            _log.warning("Langfuse tracing disabled: %s", exc)
            return None
        return _HANDLER


def flush() -> None:
    """Push buffered spans (call on shutdown; also registered via atexit)."""
    client = _CLIENT
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001
        pass


def reset_for_tests() -> None:
    global _HANDLER, _CLIENT, _INIT_FAILED
    _HANDLER = None
    _CLIENT = None
    _INIT_FAILED = False


def trace_config(
    *,
    session_id: str,
    user_id: str | None = None,
    name: str | None = None,
    tags: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RunnableConfig fragment (callbacks + ``langfuse_*`` metadata); ``{}`` when off."""
    handler = get_handler()
    if handler is None:
        return {}
    md: dict[str, Any] = {
        "langfuse_session_id": str(session_id),
        "langfuse_trace_name": name or DEFAULT_TRACE_NAME,
        "langfuse_tags": [str(t) for t in tags],
    }
    if user_id:
        md["langfuse_user_id"] = str(user_id)
    for k, v in (metadata or {}).items():
        md.setdefault(k, v)
    return {"callbacks": [handler], "metadata": md}


def with_tracing(config: dict[str, Any], **trace_kwargs: Any) -> dict[str, Any]:
    """Return *config* merged with :func:`trace_config` (callbacks appended)."""
    extra = trace_config(**trace_kwargs)
    if not extra:
        return config
    merged = dict(config)
    merged["callbacks"] = list(config.get("callbacks") or []) + list(extra["callbacks"])
    md = dict(config.get("metadata") or {})
    md.update(extra["metadata"])
    merged["metadata"] = md
    return merged
