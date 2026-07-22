"""Admin API routes: login, credits, service provider settings, hospital data views."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Cookie, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from conversation_log import (
    delete_call,
    get_call_detail,
    list_calls_for_patient,
)
from admin_credits import collect_credit_reports
from database import (
    _enrich_appointment,
    _get_doctor_by_id,
    _get_doctor_by_name,
    _get_patient_by_id,
    _load_all_appointments,
    _load_doctors,
    _load_patients,
    _load_prescriptions,
    _normalize_doctor_name,
    _normalize_phone,
    _patient_past_doctors,
    add_doctor_unavailable,
    department_matches,
    ensure_default_lunch_breaks,
    get_doctor_day_grid,
    parse_appointment_datetime,
    remove_doctor_unavailable,
)
from service_settings import (
    apply_settings_to_env,
    is_provider_enabled,
    load_settings,
    options_catalog,
    resolve_voice_pipeline,
    save_settings,
)

ADMIN_USER = os.getenv("ADMIN_USER", "Admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "12345")

_SECRET_PATH = Path(__file__).resolve().parent / ".admin_token_secret"
if os.getenv("ADMIN_TOKEN_SECRET"):
    _TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET")
elif _SECRET_PATH.exists():
    _TOKEN_SECRET = _SECRET_PATH.read_text(encoding="utf-8").strip() or secrets.token_hex(16)
else:
    _TOKEN_SECRET = secrets.token_hex(16)
    try:
        _SECRET_PATH.write_text(_TOKEN_SECRET, encoding="utf-8")
    except OSError:
        pass

_ADMIN_DIR = Path(__file__).resolve().parent / "admin"
router = APIRouter(prefix="/admin", tags=["admin"])


def _make_token(username: str) -> str:
    sig = hmac.new(
        _TOKEN_SECRET.encode(),
        username.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{username}.{sig}"


def _valid_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    user, sig = token.split(".", 1)
    expected = hmac.new(
        _TOKEN_SECRET.encode(),
        user.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(sig, expected) and user == ADMIN_USER


def _require_auth(admin_token: str | None) -> None:
    if not _valid_token(admin_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _page(name: str) -> FileResponse:
    path = _ADMIN_DIR / name
    return FileResponse(
        path,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )


def _parse_appt_time(value: str) -> Optional[datetime]:
    return parse_appointment_datetime(value)


class LoginBody(BaseModel):
    username: str
    password: str


class SettingsBody(BaseModel):
    stt: str | None = None
    tts: str | None = None
    cascade_llm: str | None = None
    cli_llm: str | None = None
    voice_pipeline_default: str | None = None
    deepgram_voice: str | None = None
    glm_model: str | None = None
    groq_model: str | None = None
    deepseek_model: str | None = None
    openai_realtime_model: str | None = None
    openai_realtime_voice: str | None = None
    debug_mode: bool | None = None
    save_llm_messages: bool | None = None
    vad_stop_secs: float | None = None
    enabled_providers: dict[str, bool] | None = None
    # Flat flags (preferred) — False values must not be dropped on save.
    enable_deepseek: bool | None = None
    enable_glm: bool | None = None
    enable_groq: bool | None = None
    enable_openai: bool | None = None
    enable_realtime: bool | None = None


# -------- Pages --------
@router.get("/")
async def admin_page():
    return _page("index.html")


@router.get("/appointments")
async def appointments_page():
    return _page("appointments.html")


@router.get("/patients")
async def patients_page():
    return _page("patients.html")


@router.get("/patient")
@router.get("/patient/{patient_id}")
async def patient_page(patient_id: str | None = None):
    return _page("patient.html")


@router.get("/doctors")
async def doctors_page():
    return _page("doctors.html")


@router.get("/doctor")
@router.get("/doctor/{doctor_id}")
async def doctor_page(doctor_id: str | None = None):
    return _page("doctor.html")


@router.get("/messages")
async def messages_page():
    return _page("messages.html")


# -------- Auth / settings --------
@router.post("/api/login")
async def login(body: LoginBody, response: Response):
    username = (body.username or "").strip()
    password = body.password or ""
    if username != ADMIN_USER or password != ADMIN_PASS:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = _make_token(username)
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 12,
        path="/",
    )
    return {"ok": True, "user": username}


@router.post("/api/logout")
async def logout(response: Response):
    response.delete_cookie("admin_token", path="/")
    return {"ok": True}


@router.get("/api/session")
async def session(admin_token: str | None = Cookie(default=None)):
    if not _valid_token(admin_token):
        return {"authenticated": False}
    return {"authenticated": True, "user": ADMIN_USER}


@router.get("/api/ui-config")
async def ui_config():
    """Public flags for the chat UI (no login)."""
    s = load_settings()
    return {
        "debug_mode": bool(s.get("debug_mode", False)),
        "voice_pipeline_default": resolve_voice_pipeline(s),
        "cascade_llm": s.get("cascade_llm") or "deepseek",
        "cli_llm": s.get("cli_llm") or "groq",
        "stt": s.get("stt") or "deepgram",
        "tts": s.get("tts") or "deepgram",
        "enabled_realtime": is_provider_enabled(s, "realtime"),
        "enabled_providers": s.get("enabled_providers") or {},
    }


@router.get("/api/credits")
async def credits(admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    reports = await collect_credit_reports()
    return {"providers": reports}


@router.get("/api/settings")
async def get_settings(admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    apply_settings_to_env()
    return {"settings": load_settings(), "options": options_catalog()}


@router.put("/api/settings")
async def put_settings(body: SettingsBody, admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    # exclude_unset keeps explicit False values (enable_openai=false, etc.).
    updates = body.model_dump(exclude_unset=True)
    if "enabled_providers" in updates and updates["enabled_providers"] is not None:
        updates["enabled_providers"] = {
            str(k): bool(v) for k, v in dict(updates["enabled_providers"]).items()
        }
    try:
        saved = save_settings(updates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "settings": saved}


# -------- Hospital data APIs --------
@router.get("/api/departments")
async def departments(admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    deps = sorted({(d.get("department") or "").strip() for d in _load_doctors() if d.get("department")})
    return {"ok": True, "departments": deps}


@router.get("/api/doctors")
async def doctors(
    department: Optional[str] = Query(default=None),
    admin_token: str | None = Cookie(default=None),
):
    _require_auth(admin_token)
    ensure_default_lunch_breaks()
    docs = _load_doctors()
    dep = (department or "").strip()
    if dep:
        docs = [d for d in docs if department_matches(dep, d.get("department") or "")]
    return {"ok": True, "count": len(docs), "doctors": docs}


class UnavailableBody(BaseModel):
    start_hm: str
    end_hm: str
    day: str | None = None
    reason: str | None = "unavailable"


@router.get("/api/doctors/{doctor_id}/schedule")
async def doctor_schedule(
    doctor_id: str,
    day: Optional[str] = Query(default=None),
    admin_token: str | None = Cookie(default=None),
):
    """8×6 day grid for one doctor (09:00–17:00, 10-minute slots)."""
    _require_auth(admin_token)
    ensure_default_lunch_breaks()
    day_key = (day or "").strip() or datetime.now().strftime("%Y-%m-%d")
    try:
        grid = get_doctor_day_grid(doctor_id, day_key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"ok": True, **grid}


@router.post("/api/doctors/{doctor_id}/unavailable")
async def doctor_unavailable_add(
    doctor_id: str,
    body: UnavailableBody,
    admin_token: str | None = Cookie(default=None),
):
    _require_auth(admin_token)
    try:
        block = add_doctor_unavailable(
            doctor_id,
            body.start_hm,
            body.end_hm,
            day=(body.day or "").strip(),
            reason=(body.reason or "unavailable").strip() or "unavailable",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "block": block}


@router.delete("/api/doctors/{doctor_id}/unavailable/{block_id}")
async def doctor_unavailable_delete(
    doctor_id: str,
    block_id: str,
    admin_token: str | None = Cookie(default=None),
):
    _require_auth(admin_token)
    if not remove_doctor_unavailable(block_id):
        raise HTTPException(status_code=404, detail="Block not found")
    return {"ok": True, "removed": block_id}


@router.get("/api/appointments")
async def appointments(
    department: Optional[str] = Query(default=None),
    doctor_id: Optional[str] = Query(default=None),
    doctor: Optional[str] = Query(default=None),
    from_now: bool = Query(default=True),
    include_cancelled: bool = Query(default=False),
    admin_token: str | None = Cookie(default=None),
):
    """Future (or all) appointments, filterable by department / doctor."""
    _require_auth(admin_token)
    now = datetime.now()
    dep = (department or "").strip().lower()
    doc_id = (doctor_id or "").strip().upper()
    doc_name = _normalize_doctor_name(doctor) if doctor else ""

    rows: list[dict[str, Any]] = []
    for a in _load_all_appointments():
        if not include_cancelled and a.get("status") == "CANCELLED":
            continue

        enriched = _enrich_appointment(a)
        a_dep = (enriched.get("department") or "").strip().lower()
        a_doc_id = str(enriched.get("doctor_id") or "").strip().upper()
        a_doc_name = _normalize_doctor_name(str(enriched.get("doctor") or ""))

        if dep and not department_matches(dep, a_dep):
            continue
        if doc_id and a_doc_id != doc_id:
            continue
        if doc_name and a_doc_name != doc_name:
            continue

        when = _parse_appt_time(str(enriched.get("time") or ""))
        if from_now:
            if when is None or when < now:
                continue

        enriched["_sort_time"] = when.isoformat() if when else ""
        rows.append(enriched)

    rows.sort(key=lambda r: r.get("_sort_time") or r.get("time") or "")
    for r in rows:
        r.pop("_sort_time", None)

    return {"ok": True, "count": len(rows), "appointments": rows, "as_of": now.strftime("%Y-%m-%d %H:%M")}


@router.get("/api/patients")
async def patients(
    q: Optional[str] = Query(default=None),
    admin_token: str | None = Cookie(default=None),
):
    _require_auth(admin_token)
    query = (q or "").strip().lower()
    rows = _load_patients()
    if query:
        phone_q = _normalize_phone(query)
        filtered = []
        for p in rows:
            hay = " ".join(
                [
                    str(p.get("patient_id") or ""),
                    str(p.get("name") or ""),
                    str(p.get("phone") or ""),
                    str(p.get("address") or ""),
                ]
            ).lower()
            if query in hay or (phone_q and phone_q in _normalize_phone(str(p.get("phone") or ""))):
                filtered.append(p)
        rows = filtered
    rows = sorted(rows, key=lambda p: str(p.get("name") or "").lower())
    return {"ok": True, "count": len(rows), "patients": rows}


@router.get("/api/patients/{patient_id}")
async def patient_detail(patient_id: str, admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    patient = _get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient not found: {patient_id}")

    visits = []
    for a in _load_all_appointments():
        if a.get("patient_id") != patient.get("patient_id"):
            continue
        visits.append(_enrich_appointment(a))
    visits.sort(key=lambda v: str(v.get("time") or ""), reverse=True)

    prescriptions = []
    for r in _load_prescriptions():
        if r.get("patient_id") != patient.get("patient_id"):
            continue
        item = dict(r)
        doctor = _get_doctor_by_id(str(r.get("doctor_id") or ""))
        if doctor is None and r.get("doctor"):
            doctor = _get_doctor_by_name(str(r.get("doctor")))
        if doctor:
            item["doctor_name"] = doctor.get("name", "")
            item["department"] = doctor.get("department", "")
            item["doctor_id"] = doctor.get("doctor_id", item.get("doctor_id"))
        prescriptions.append(item)

    return {
        "ok": True,
        "patient": patient,
        "visits": visits,
        "visit_count": len(visits),
        "prescriptions": prescriptions,
        "prescription_count": len(prescriptions),
        "past_doctors": _patient_past_doctors(str(patient.get("patient_id"))),
        "chats": list_calls_for_patient(str(patient.get("patient_id"))),
    }


@router.get("/api/patients/{patient_id}/chats")
async def patient_chats(patient_id: str, admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    patient = _get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient not found: {patient_id}")
    chats = list_calls_for_patient(str(patient.get("patient_id")))
    return {"ok": True, "patient_id": patient.get("patient_id"), "count": len(chats), "chats": chats}


@router.get("/api/chats/{session_id}")
async def chat_detail(session_id: str, admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    detail = get_call_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Chat not found: {session_id}")
    return {"ok": True, "chat": detail}


@router.get("/api/response-timings")
async def response_timings(admin_token: str | None = Cookie(default=None)):
    """Average first-text / first-speech latency across all chats/sessions."""
    _require_auth(admin_token)
    from conversation_log import summarize_response_timings

    return summarize_response_timings()


@router.delete("/api/chats/{session_id}")
async def chat_delete(session_id: str, admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    if not delete_call(session_id):
        raise HTTPException(status_code=404, detail=f"Chat not found: {session_id}")
    return {"ok": True, "deleted": session_id}


@router.get("/api/llm-messages/sessions")
async def llm_message_sessions(admin_token: str | None = Cookie(default=None)):
    """List LLM dump sessions — oldest first for analysis."""
    _require_auth(admin_token)
    from llm_message_dump import list_llm_message_sessions

    sessions = list_llm_message_sessions(oldest_first=True)
    return {"ok": True, "count": len(sessions), "sessions": sessions}


@router.delete("/api/llm-messages/sessions")
@router.post("/api/llm-messages/delete-all")
async def llm_message_sessions_delete_all(admin_token: str | None = Cookie(default=None)):
    _require_auth(admin_token)
    from llm_message_dump import delete_all_llm_messages

    removed = delete_all_llm_messages()
    return {"ok": True, "deleted": removed}


@router.delete("/api/llm-messages/sessions/{session_id}")
@router.post("/api/llm-messages/sessions/{session_id}/delete")
async def llm_message_session_delete(
    session_id: str, admin_token: str | None = Cookie(default=None)
):
    _require_auth(admin_token)
    from llm_message_dump import delete_session_messages

    # Idempotent: missing folder still counts as deleted for the UI.
    delete_session_messages(session_id)
    return {"ok": True, "deleted": session_id}


@router.get("/api/llm-messages/sessions/{session_id}")
async def llm_message_session_detail(
    session_id: str, admin_token: str | None = Cookie(default=None)
):
    _require_auth(admin_token)
    from llm_message_dump import list_session_pairs, read_session_runtime_meta

    pairs = list_session_pairs(session_id)
    if not pairs:
        raise HTTPException(status_code=404, detail=f"No LLM messages for session: {session_id}")
    return {
        "ok": True,
        "session_id": session_id,
        "count": len(pairs),
        "pairs": pairs,
        "session_meta": read_session_runtime_meta(session_id),
    }


@router.get("/api/llm-messages/sessions/{session_id}/pairs/{n}")
async def llm_message_pair(
    session_id: str, n: int, admin_token: str | None = Cookie(default=None)
):
    _require_auth(admin_token)
    from llm_message_dump import list_session_pairs, read_session_pair

    pair = read_session_pair(session_id, n)
    if not pair:
        raise HTTPException(status_code=404, detail=f"Pair not found: {session_id} #{n}")
    pairs = list_session_pairs(session_id)
    nums = [int(p.get("n") or 0) for p in pairs]
    idx = nums.index(n) if n in nums else -1
    return {
        "ok": True,
        "pair": pair,
        "index": idx,
        "total": len(pairs),
        "prev_n": nums[idx - 1] if idx > 0 else None,
        "next_n": nums[idx + 1] if 0 <= idx < len(nums) - 1 else None,
    }
