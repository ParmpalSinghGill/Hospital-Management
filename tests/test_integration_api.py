"""Integration: FastAPI admin / tools / call-log HTTP APIs."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def api_app(isolated_db, isolated_chats, monkeypatch):
    import admin_routes
    import log_routes
    import tool_routes

    monkeypatch.setattr(admin_routes, "ADMIN_USER", "Admin")
    monkeypatch.setattr(admin_routes, "ADMIN_PASS", "test-secret")

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.include_router(log_routes.router)
    app.include_router(tool_routes.router)
    return app


@pytest.fixture
def client(api_app):
    return TestClient(api_app)


@pytest.fixture
def authed(client):
    res = client.post(
        "/admin/api/login",
        json={"username": "Admin", "password": "test-secret"},
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True
    return client


def test_login_rejects_bad_password(client):
    res = client.post(
        "/admin/api/login",
        json={"username": "Admin", "password": "wrong"},
    )
    assert res.status_code == 401


def test_admin_doctors_and_departments(authed):
    deps = authed.get("/admin/api/departments")
    assert deps.status_code == 200
    assert "Cardiology" in deps.json()["departments"]

    docs = authed.get("/admin/api/doctors", params={"department": "Cardiology"})
    assert docs.status_code == 200
    body = docs.json()
    assert body["ok"] and body["count"] >= 1

    did = body["doctors"][0]["doctor_id"]
    grid = authed.get(f"/admin/api/doctors/{did}/schedule", params={"day": "2026-12-20"})
    assert grid.status_code == 200
    assert grid.json()["stats"]["total"] == 48


def test_admin_patients_and_appointments_emptyish(authed):
    patients = authed.get("/admin/api/patients")
    assert patients.status_code == 200
    assert patients.json()["ok"] is True

    appts = authed.get("/admin/api/appointments", params={"from_now": False})
    assert appts.status_code == 200
    assert appts.json()["ok"] is True


def test_admin_settings_cost_controls_persist(authed, tmp_path, monkeypatch):
    import service_settings as ss

    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)

    res = authed.put(
        "/admin/api/settings",
        json={
            "enable_deepseek": True,
            "enable_glm": True,
            "enable_groq": True,
            "enable_openai": False,
            "enable_realtime": False,
            "cascade_llm": "deepseek",
            "cli_llm": "groq",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["settings"]["enable_openai"] is False
    assert body["settings"]["enable_realtime"] is False
    assert body["settings"]["enabled_providers"]["openai"] is False

    again = authed.get("/admin/api/settings")
    assert again.status_code == 200
    settings = again.json()["settings"]
    assert settings["enable_openai"] is False
    assert settings["enable_realtime"] is False

    ui = authed.get("/admin/api/ui-config")
    assert ui.status_code == 200
    assert ui.json()["enabled_realtime"] is False


def test_tool_discovery_api(client):
    catalog = client.get("/api/tools")
    assert catalog.status_code == 200
    data = catalog.json()
    assert data["ok"] and data["count"] == 7

    one = client.get("/api/tools/book_appointment")
    assert one.status_code == 200
    assert one.json()["tool"]["name"] == "book_appointment"

    missing = client.get("/api/tools/not_a_tool")
    assert missing.status_code == 404

    html = client.get("/toollist/")
    assert html.status_code == 200
    assert "book_appointment" in html.text


def test_call_log_http_lifecycle(client, isolated_chats):
    start = client.post(
        "/api/call-logs/start",
        json={
            "call_id": "http-call-001",
            "session_id": "http-call-001",
            "pipeline_mode": "cli",
            "channel": "web_app",
        },
    )
    assert start.status_code == 200
    assert start.json()["ok"] is True

    event = client.post(
        "/api/call-logs/event",
        json={
            "call_id": "http-call-001",
            "type": "user_text",
            "text": "I need a dentist",
        },
    )
    assert event.status_code == 200
    assert event.json()["ok"] is True

    end = client.post("/api/call-logs/end", json={"call_id": "http-call-001"})
    assert end.status_code == 200

    recent = client.get("/api/call-logs/recent", params={"limit": 10})
    assert recent.status_code == 200
    sessions = recent.json()["sessions"]
    assert any(s.get("session_id") == "http-call-001" for s in sessions)
