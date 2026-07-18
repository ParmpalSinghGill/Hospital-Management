"""HTTP API for per-session conversation JSON logs (client timings + listing)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from conversation_log import (
    apply_client_event,
    end_call,
    list_recent_calls,
    start_call,
)

router = APIRouter(prefix="/api/call-logs", tags=["call-logs"])


class StartBody(BaseModel):
    call_id: str = Field(min_length=4, max_length=120)
    session_id: str = ""
    user_id: str = ""
    pipeline_mode: str = ""
    channel: str = "web_app"
    audio_codec: str = "opus"


class EventBody(BaseModel):
    call_id: str = Field(min_length=4, max_length=120)
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


@router.post("/start")
async def start_log(body: StartBody) -> dict[str, Any]:
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
    }


@router.post("/event")
async def log_event(body: EventBody) -> dict[str, Any]:
    try:
        ix = apply_client_event(body.call_id, body.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "turn_number": ix.get("turn_number"),
        "mode": ix.get("mode"),
        "complexity_metrics": ix.get("complexity_metrics"),
    }


@router.post("/end")
async def end_log(body: EndBody) -> dict[str, Any]:
    data = end_call(body.call_id)
    return {"ok": True, "session_end_time": (data or {}).get("session_end_time")}


@router.get("/recent")
async def recent(limit: int = 20) -> dict[str, Any]:
    return {"sessions": list_recent_calls(limit=min(max(limit, 1), 100))}
