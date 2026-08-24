"""HIPAA technical-safeguard tests: auth, audit, redaction, identity, retention, backups."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ----------------------------------------------------------------- fixtures ----
@pytest.fixture
def secure_env(tmp_path, monkeypatch):
    """HIPAA mode on, all PHI stores + secrets under a temp DATA_DIR."""
    monkeypatch.setenv("HIPAA_MODE", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADMIN_TOKEN_SECRET", "unit-test-secret-0123456789abcdef")
    monkeypatch.delenv("ADMIN_USER", raising=False)
    monkeypatch.delenv("ADMIN_PASS", raising=False)
    monkeypatch.delenv("PHI_ALLOWED_PROVIDERS", raising=False)
    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "0")  # local .env may set it; tests pin production posture
    import security_config as sc

    sc._SECRET_CACHE = None
    import admin_auth
    from rate_limit import Lockout, RateLimiter

    admin_auth.revoke_all_sessions()
    monkeypatch.setattr(admin_auth, "_LOCKOUT", Lockout(max_failures=5, window_secs=900, lock_secs=900))
    monkeypatch.setattr(admin_auth, "_LOGIN_RATE", RateLimiter(max_events=20, window_secs=60))
    import audit_log

    audit_log._LAST_HASH = None
    yield tmp_path / "data"


# ------------------------------------------------------------------ config -----
def test_hipaa_mode_blocks_non_baa_providers(secure_env, monkeypatch):
    import security_config as sc

    assert sc.hipaa_mode()
    assert sc.allowed_llm_providers() == frozenset({"openai"})
    assert not sc.provider_allowed("deepseek")
    assert not sc.provider_allowed("glm")
    monkeypatch.setenv("PHI_ALLOWED_PROVIDERS", "openai,groq,deepseek,glm")
    # never-PHI vendors cannot be allow-listed even explicitly
    assert sc.allowed_llm_providers() == frozenset({"openai", "groq"})


def test_model_init_refuses_blocked_provider(secure_env, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    import Model

    with pytest.raises(RuntimeError, match="blocked"):
        Model._init_model(callbacks=None, provider="deepseek")


def test_settings_resolve_to_allowed_llm(secure_env, monkeypatch, tmp_path):
    import service_settings as ss

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "admin_settings.json")
    s = ss.load_settings()
    assert s["cascade_llm"] == "openai"
    assert s["cli_llm"] == "openai"
    assert s["save_llm_messages"] is False
    cat = ss.options_catalog(s)
    blocked = {o["id"]: o for o in cat["cascade_llm"]}
    assert blocked["deepseek"]["blocked"] and not blocked["deepseek"]["enabled"]
    assert blocked["openai"]["enabled"]


def test_non_phi_channels_handoff(secure_env, monkeypatch):
    import security_config as sc

    assert not sc.channel_may_carry_phi("telegram")
    assert not sc.channel_may_carry_phi("discord")
    assert sc.channel_may_carry_phi("web_app")
    monkeypatch.setenv("HIPAA_MODE", "0")
    assert sc.channel_may_carry_phi("telegram")


# -------------------------------------------------------------- redaction -----
def test_redaction_masks_phone_name_and_text():
    from phi_redact import mask_name, mask_phone, redact_obj, redact_text

    assert mask_phone("+91 98765 43210") == "**********10"
    assert mask_name("Aadhya Iyer") == "A***** I***"
    out = redact_text("call me on 9876543210 or a@b.com")
    assert "9876543210" not in out and "a@b.com" not in out
    red = redact_obj({"phone": "9876543210", "patient_name": "Aadhya Iyer", "address": "12 Main",
                      "patient_id": "PAT-1", "ok": True, "nested": {"name": "X Y"}})
    assert red["patient_id"] == "PAT-1" and red["ok"] is True
    assert red["phone"].endswith("10") and "9876543210" not in json.dumps(red)
    assert red["address"] == "[redacted]"
    assert red["nested"]["name"] == "X Y"  # single letters have nothing to mask


def test_log_redaction_filter(secure_env, caplog):
    import logging

    from phi_redact import install_log_redaction

    install_log_redaction()
    logging.getLogger().addFilter  # filters exist
    logger = logging.getLogger("t")
    with caplog.at_level(logging.INFO):
        logger.info("patient phone 9876543210 booked")
    # root filter redacts message text
    import phi_redact

    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "phone 9876543210", (), None)
    assert phi_redact._RedactingFilter().filter(rec) is True
    assert "9876543210" not in rec.getMessage()


# ------------------------------------------------------------------ audit -----
def test_audit_chain_appends_and_verifies(secure_env):
    import audit_log

    audit_log.audit("test.one", actor="u1", subject=audit_log.subject_ref(patient_id="PAT-1", phone="9876543210"))
    audit_log.audit("test.two", actor="u1", detail={"n": 2})
    rows = list(audit_log.iter_records())
    assert [r["event"] for r in rows] == ["test.one", "test.two"]
    assert "9876543210" not in audit_log.audit_path().read_text()
    assert rows[0]["subject"]["patient_id"] == "PAT-1" and "phone_hash" in rows[0]["subject"]
    assert audit_log.verify_chain()["ok"] is True
    # tamper → chain breaks
    p = audit_log.audit_path()
    lines = p.read_text().splitlines()
    lines[0] = lines[0].replace("test.one", "test.xxx")
    p.write_text("\n".join(lines) + "\n")
    assert audit_log.verify_chain()["ok"] is False
    assert oct(p.stat().st_mode & 0o777) == "0o600"


# -------------------------------------------------------------- admin auth ----
def test_admin_rejects_weak_env_password_in_hipaa_mode(secure_env, monkeypatch):
    import admin_auth

    monkeypatch.setenv("ADMIN_USER", "Admin")
    monkeypatch.setenv("ADMIN_PASS", "12345")
    assert not admin_auth.admin_configured()
    with pytest.raises(admin_auth.LoginError) as exc:
        admin_auth.login("Admin", "12345")
    assert exc.value.status == 503


def test_admin_login_lockout_and_session_lifecycle(secure_env):
    import admin_auth

    admin_auth.upsert_user("ops", "correct-horse-battery", role="admin")
    admin_auth.upsert_user("ro", "viewer-passphrase-1", role="viewer")
    # hash on disk, not the password
    raw = Path(admin_auth._users_path()).read_text()
    assert "correct-horse-battery" not in raw and "scrypt$" in raw
    assert oct(Path(admin_auth._users_path()).stat().st_mode & 0o777) == "0o600"

    user, token = admin_auth.login("ops", "correct-horse-battery", client_key="1.2.3.4")
    assert user.role == "admin"
    sess = admin_auth.resolve_session(token)
    assert sess and sess.username == "ops"
    # forged signature fails
    assert admin_auth.resolve_session(token[:-4] + "zzzz") is None
    # revoke
    admin_auth.revoke_session(token)
    assert admin_auth.resolve_session(token) is None

    # lockout after 5 failures
    for _ in range(5):
        with pytest.raises(admin_auth.LoginError):
            admin_auth.login("ro", "wrong", client_key="5.6.7.8")
    with pytest.raises(admin_auth.LoginError) as exc:
        admin_auth.login("ro", "viewer-passphrase-1", client_key="5.6.7.8")
    assert exc.value.status == 423


def test_admin_http_role_gating_and_audit(secure_env, isolated_db, isolated_chats, tmp_path):
    import admin_auth
    import admin_routes
    import log_routes

    admin_auth.upsert_user("ops", "correct-horse-battery", role="admin")
    admin_auth.upsert_user("ro", "viewer-passphrase-1", role="viewer")
    app = FastAPI()
    app.include_router(admin_routes.router)
    app.include_router(log_routes.router)
    # Secure cookies are on in HIPAA mode, so talk to the app over https like a real deployment.
    c = TestClient(app, base_url="https://testserver")

    # unauthenticated HTML pages redirect to login
    r = c.get("/admin/patients", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/admin/"

    assert c.get("/admin/api/patients").status_code == 401
    assert c.get("/api/call-logs/recent").status_code == 401

    # viewer can read, cannot write
    c.post("/admin/api/login", json={"username": "ro", "password": "viewer-passphrase-1"})
    assert c.get("/admin/api/patients").status_code == 200
    assert c.put("/admin/api/settings", json={"debug_mode": False}).status_code == 403
    assert c.get("/admin/api/audit").status_code == 403
    c.post("/admin/api/logout")
    assert c.get("/admin/api/patients").status_code == 401

    # admin: may write, but cannot enable raw LLM dumps in HIPAA mode
    c.post("/admin/api/login", json={"username": "ops", "password": "correct-horse-battery"})
    assert c.put("/admin/api/settings", json={"save_llm_messages": True}).status_code == 400
    aud = c.get("/admin/api/audit").json()
    events = [r["event"] for r in aud["records"]]
    assert "admin.login" in events and "admin.patients.list" in events
    assert aud["chain"]["ok"] is True


# --------------------------------------------------------------- identity -----
def test_identity_flow_never_discloses_record(isolated_db, secure_env):
    from Tools import book_appointment, cancel_appointment, get_prescriptions, lookup_patient, save_patient

    def j(t, **kw):
        return json.loads(t.invoke(kw))

    assert j(save_patient, patient_name="Priya Nair", phone="9900000001")["ok"]
    # phone only: no record, no name
    r = j(lookup_patient, phone="9900000001")
    assert r["needs_name"] is True and "Priya" not in json.dumps(r)
    # wrong name: no hint
    r = j(lookup_patient, phone="9900000001", patient_name="Someone Else")
    assert r["name_mismatch"] is True and "Priya" not in json.dumps(r)
    # patient id alone is not enough
    r = j(lookup_patient, patient_id="PAT-0001")
    assert r["ok"] is False and r["needs_phone"] is True
    # collision on save does not reveal the other person
    r = j(save_patient, patient_name="Other Name", phone="9900000001")
    assert r["ok"] is False and r.get("phone_conflict") and "Priya" not in json.dumps(r)
    r = j(book_appointment, patient_name="Other Name", phone="9900000001", doctor="Dr. Test Cardio", time="2026-12-20 10:00")
    assert r["ok"] is False and "Priya" not in json.dumps(r)
    # prescriptions need phone + name
    r = j(get_prescriptions, patient_id="PAT-0001")
    assert r["ok"] is False and r["needs_phone"] and r["needs_name"]
    # cancel requires phone
    r = j(cancel_appointment, appointment_id="APT-0001")
    assert r["ok"] is False and r["needs_phone"]


def test_identity_lockout_after_repeated_mismatch(isolated_db, secure_env):
    import tool_patient
    from Tools import lookup_patient, save_patient

    tool_patient._MISMATCH_LIMIT.reset("anon")
    tool_patient._LOOKUP_LIMIT.reset("anon")
    assert json.loads(save_patient.invoke({"patient_name": "Kiran Rao", "phone": "9900000002"}))["ok"]
    out = None
    for i in range(4):
        out = json.loads(lookup_patient.invoke({"phone": "9900000002", "patient_name": f"Wrong {i}"}))
    assert out["name_mismatch"] is True and out.get("locked") is True


# ------------------------------------------------------- retention/backup -----
def test_retention_purges_old_sessions_and_keeps_new(secure_env, monkeypatch):
    import security_config as sc
    import retention

    chats = sc.chats_dir()
    (chats / "sessions").mkdir(parents=True, exist_ok=True)
    old = chats / "sessions" / "sess-20200101-000000_old.json"
    new = chats / "sessions" / "sess-20990101-000000_new.json"
    old.write_text("{}")
    new.write_text("{}")
    past = time.time() - 400 * 86400
    os.utime(old, (past, past))
    monkeypatch.setenv("RETAIN_SESSIONS_DAYS", "90")
    dry = retention.run_retention(dry_run=True)
    assert dry["sessions"] == 1 and old.exists()
    res = retention.run_retention()
    assert res["sessions"] == 1 and not old.exists() and new.exists()


def test_backup_create_verify_restore_encrypted(secure_env, isolated_db, monkeypatch):
    from cryptography.fernet import Fernet

    import backup

    monkeypatch.setenv("BACKUP_ENC_KEY", Fernet.generate_key().decode())
    made = backup.create_backup()
    assert made.suffix == ".enc" and made.exists()
    assert oct(made.stat().st_mode & 0o777) == "0o600"
    assert b"SQLite format" not in made.read_bytes()[:64]
    assert backup.verify_backup(made) is True
    target = made.parent / "restored.db"
    backup.restore_backup(made, target=target)
    assert target.read_bytes()[:15] == b"SQLite format 3"


# ----------------------------------------------------------------- MCP auth ----
def test_mcp_bearer_middleware_blocks_without_key():
    pytest.importorskip("mcp")
    import anyio

    from mcp_server import _BearerAuthMiddleware

    seen = {}

    async def inner(scope, receive, send):
        seen["called"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = _BearerAuthMiddleware(inner, "sekrit")

    async def run(headers):
        out = []

        async def send(msg):
            out.append(msg)

        async def receive():
            return {"type": "http.request"}

        await mw({"type": "http", "path": "/mcp", "headers": headers, "method": "POST"}, receive, send)
        return out[0]["status"]

    assert anyio.run(run, []) == 401
    assert "called" not in seen
    assert anyio.run(run, [(b"authorization", b"Bearer sekrit")]) == 200


# --------------------------------------------------- synthetic-data posture ----
def test_synthetic_data_only_keeps_controls_but_relaxes_vendors(secure_env, monkeypatch):
    import security_config as sc

    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "1")
    assert sc.hipaa_mode() and sc.synthetic_data_only()
    assert sc.allowed_llm_providers() == frozenset({"deepseek", "glm", "groq", "openai"})
    assert sc.channel_may_carry_phi("telegram")
    # the other controls are unchanged; raw LLM dumps are allowed (synthetic data, QA needs them)
    assert sc.cookie_secure() and sc.debug_dumps_allowed()
    assert any("SYNTHETIC_DATA_ONLY" in w for w in sc.startup_warnings())
    # turning the flag off locks it back down
    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "0")
    assert sc.allowed_llm_providers() == frozenset({"openai"})
    assert not sc.channel_may_carry_phi("telegram")
    assert not sc.debug_dumps_allowed()
    monkeypatch.setenv("LLM_QA_DUMPS", "1")
    assert sc.debug_dumps_allowed()  # explicit, audited QA exception


def test_go_live_check_blocks_synthetic_posture_and_passes_when_ready(secure_env, monkeypatch, tmp_path):
    import admin_auth
    import go_live_check

    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    res = {r["check"]: r["status"] for r in go_live_check.run_checks()}
    assert res["SYNTHETIC_DATA_ONLY off"] == "FAIL"
    assert res["DEEPSEEK_API_KEY absent"] == "FAIL"
    assert go_live_check.main([]) == 1

    # now the production posture
    for k in ("SYNTHETIC_DATA_ONLY", "DEEPSEEK_API_KEY", "GLM_API_KEY", "GOOGLE_API_KEY", "ADMIN_PASS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PHI_ALLOWED_PROVIDERS", "openai")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CONFIRMED_BAAS", "openai,deepgram")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://assistant.example.org")
    monkeypatch.setenv("BACKUP_ENC_KEY", "k")
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    import security_config as sc

    sc.harden_permissions()
    admin_auth.upsert_user("ops", "correct-horse-battery", role="admin")
    import service_settings as ss

    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "admin_settings.json")
    res = {r["check"]: r for r in go_live_check.run_checks()}
    fails = [k for k, r in res.items() if r["status"] == "FAIL"]
    assert fails == [], fails
    assert go_live_check.main([]) == 0


def test_mcp_bearer_middleware_refuses_websocket():
    pytest.importorskip("mcp")
    import anyio

    from mcp_server import _BearerAuthMiddleware

    async def inner(scope, receive, send):  # pragma: no cover - must not be reached
        raise AssertionError("websocket must not reach the app")

    mw = _BearerAuthMiddleware(inner, "sekrit")

    async def run():
        out = []

        async def send(msg):
            out.append(msg)

        async def receive():
            return {"type": "websocket.connect"}

        await mw({"type": "websocket", "path": "/ws", "headers": []}, receive, send)
        return out[0]

    msg = anyio.run(run)
    assert msg == {"type": "websocket.close", "code": 1008}


# ------------------------------------------------- stdout prompt-leak (R2-02) ----
def test_llm_inspect_handler_never_dumps_with_real_phi(secure_env, monkeypatch):
    """LLMInspectHandler prints raw prompts/responses to stdout; it must be off
    whenever real PHI can be in play (HIPAA on and SYNTHETIC_DATA_ONLY off)."""
    import importlib

    import Model

    # real-PHI production posture, flag explicitly on → still refused
    monkeypatch.setenv("HIPAA_MODE", "1")
    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "0")
    monkeypatch.setenv("LLM_INSPECT_STDOUT", "1")
    importlib.reload(Model)
    assert Model.inspect_handler() is None

    # synthetic-dev posture, flag on → allowed (dev convenience)
    monkeypatch.setenv("SYNTHETIC_DATA_ONLY", "1")
    assert Model.inspect_handler() is Model.LLMInspectHandler

    # flag off → never
    monkeypatch.setenv("LLM_INSPECT_STDOUT", "0")
    assert Model.inspect_handler() is None

    # default _init_model no longer attaches a stdout dumper
    import inspect as _inspect
    assert _inspect.signature(Model._init_model).parameters["callbacks"].default is None
