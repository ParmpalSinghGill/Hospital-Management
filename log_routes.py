"""HTTP API for per-session conversation JSON logs (client timings + listing).

Security model
--------------
* ``POST /start`` is rate-limited per client IP and returns a ``call_token``
  bound to the ``call_id`` (HMAC with the server secret). The browser must
  send that token on ``/event`` and ``/end`` — nobody else can write into a
  session they did not open.
* ``GET /recent`` returns PHI (patient ids, previews) and therefore requires an
  authenticated admin session (same cookie as the Admin console).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

from fastapi import APIRouter, Cookie, Header, HTTPException, Request
from pydantic import BaseModel, Field

import security_config as sc
from audit_log import audit
from conversation_log import (
    apply_client_event,
    end_call,
    list_recent_calls,
    start_call,
)
from rate_limit import RateLimiter

router = APIRouter(prefix="/api/call-logs", tags=["call-logs"])

_START_LIMIT = RateLimiter(max_events=30, window_secs=60)
_EVENT_LIMIT = RateLimiter(max_events=600, window_secs=60)


def call_token_for(call_id: str) -> str:
    return hmac.new(sc.server_secret().encode(), f"call:{call_id}".encode(), hashlib.sha256).hexdigest()[:40]


def _check_call_token(call_id: str, token: str | None) -> None:
    if not token or not hmac.compare_digest(token, call_token_for(call_id)):
        raise HTTPException(status_code=403, detail="Invalid or missing call_token for this call_id")


def _client_key(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _require_admin(admin_token: str | None) -> str:
    from admin_auth import resolve_session

    sess = resolve_session(admin_token)
    if not sess:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return sess.username


class StartBody(BaseModel):
    call_id: str = Field(min_length=4, max_length=120)
    session_id: str = ""
    user_id: str = ""
    pipeline_mode: str = ""
    channel: str = "web_app"
    audio_codec: str = "opus"


class EventBody(BaseModel):
    call_id: str = Field(min_length=4, max_length=120)
    call_token: str | None = None
    type: str
    user_id: str | None = None
    pipeline_mode: str | None = None
    channel: str | None = None
    text: str | None = None
    sent_at_ms: float | None = None
    at_ms: float | None = None
    voice_start_at_ms: float | None = None
    voice_end_at_ms: float | None = None


class EndBody(BaseModel):
    call_id: str = Field(min_length=4, max_length=120)
    call_token: str | None = None


@router.post("/start")
async def start_log(body: StartBody, request: Request) -> dict[str, Any]:
    if not _START_LIMIT.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many sessions started; slow down")
    sid = body.session_id or body.call_id
    data = start_call(
        sid,
        pipeline_mode=body.pipeline_mode,
        session_id=sid,
        channel=body.channel,
        user_id=body.user_id,
        audio_codec=body.audio_codec,
    )
    return {
        "ok": True,
        "session_id": data.get("session_id"),
        "session_start_time": data.get("session_start_time"),
        "call_token": call_token_for(sid),
    }


@router.post("/event")
async def log_event(
    body: EventBody,
    request: Request,
    x_call_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_call_token(body.call_id, body.call_token or x_call_token)
    if not _EVENT_LIMIT.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="Too many events; slow down")
    payload = body.model_dump()
    payload.pop("call_token", None)
    try:
        ix = apply_client_event(body.call_id, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "turn_number": ix.get("turn_number"),
        "mode": ix.get("mode"),
        "complexity_metrics": ix.get("complexity_metrics"),
    }


@router.post("/end")
async def end_log(body: EndBody, x_call_token: str | None = Header(default=None)) -> dict[str, Any]:
    _check_call_token(body.call_id, body.call_token or x_call_token)
    data = end_call(body.call_id)
    return {"ok": True, "session_end_time": (data or {}).get("session_end_time")}


@router.get("/recent")
async def recent(limit: int = 20, admin_token: str | None = Cookie(default=None)) -> dict[str, Any]:
    user = _require_admin(admin_token)
    rows = list_recent_calls(limit=min(max(limit, 1), 100))
    audit("admin.calls.recent", actor=user, detail={"count": len(rows)})
    return {"sessions": rows}
