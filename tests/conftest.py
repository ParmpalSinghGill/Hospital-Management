"""Shared fixtures: isolated SQLite DB and chat log directories."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point the data layer at a fresh temp SQLite file and seed two doctors."""
    import db_core
    import db_doctors

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    db_path = dataset / "hospital.db"

    monkeypatch.setattr(db_core, "DATASET_DIR", str(dataset))
    monkeypatch.setattr(db_core, "DB_PATH", str(db_path))
    monkeypatch.setattr(db_core, "DOCTORS_DB", str(dataset / "doctors.json"))
    monkeypatch.setattr(db_core, "PATIENTS_DB", str(dataset / "patients.json"))
    monkeypatch.setattr(db_core, "APPOINTMENTS_DB", str(dataset / "appointments.json"))
    monkeypatch.setattr(db_core, "PRESCRIPTIONS_DB", str(dataset / "prescriptions.json"))
    monkeypatch.setattr(db_core, "_initialized", False)

    db_core.init_db()
    db_doctors._save_doctors(
        [
            {
                "doctor_id": "DOC-0001",
                "name": "Dr. Test Cardio",
                "department": "Cardiology",
            },
            {
                "doctor_id": "DOC-0002",
                "name": "Dr. Test Dental",
                "department": "Dentistry",
            },
            {
                "doctor_id": "DOC-0003",
                "name": "Dr. Test Gastro",
                "department": "Gastroenterology",
            },
        ]
    )
    db_doctors.ensure_default_lunch_breaks()
    yield db_path


@pytest.fixture
def isolated_chats(tmp_path, monkeypatch):
    """Redirect chat session / tool-call logs to a temp directory."""
    import chat_log_core

    chats = tmp_path / "chats"
    sessions = chats / "sessions"
    sessions.mkdir(parents=True)

    monkeypatch.setattr(chat_log_core, "CHATS_DIR", chats)
    monkeypatch.setattr(chat_log_core, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(chat_log_core, "TOOL_CALL_PATH", chats / "tool_call.json")
    monkeypatch.setattr(chat_log_core, "TIMELINE_PATH", chats / "timeline.jsonl")
    monkeypatch.setattr(chat_log_core, "_MIGRATED", False)
    chat_log_core._OPEN.clear()
    yield chats
