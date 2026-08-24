"""Admin settings changes must take effect without a process restart."""
from __future__ import annotations

import json

import pytest

import service_settings as ss


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)
    monkeypatch.setenv("HIPAA_MODE", "0")
    return path


def test_shared_graph_rebuilds_on_provider_or_model_change(settings_file, monkeypatch):
    vb = pytest.importorskip("voice_bridge")
    builds: list[str] = []

    def fake_create(*, llm_provider=None):
        builds.append(f"{llm_provider}:{ss.llm_model_for(ss.load_settings(), llm_provider)}")
        return object()

    monkeypatch.setattr(vb, "create_hospital_graph", fake_create)
    monkeypatch.setattr(vb, "_SHARED_GRAPH", None)
    monkeypatch.setattr(vb, "_SHARED_GRAPH_PROVIDER", None)
    monkeypatch.setattr(vb, "_SHARED_GRAPH_MODEL", None)

    ss.save_settings({"cascade_llm": "deepseek", "deepseek_model": "deepseek-chat"})
    g1 = vb.get_shared_hospital_graph()
    assert vb.get_shared_hospital_graph() is g1          # unchanged settings → cached
    assert builds == ["deepseek:deepseek-chat"]

    ss.save_settings({"deepseek_model": "deepseek-v4-flash"})  # same provider, new model
    g2 = vb.get_shared_hospital_graph()
    assert g2 is not g1
    assert builds[-1] == "deepseek:deepseek-v4-flash"

    ss.save_settings({"cascade_llm": "openai", "openai_chat_model": "gpt-5.6-luna"})
    g3 = vb.get_shared_hospital_graph()
    assert g3 is not g2
    assert builds[-1] == "openai:gpt-5.6-luna"


def test_ui_config_reflects_saved_settings_without_restart(settings_file):
    ss.save_settings({"cascade_llm": "openai", "openai_chat_model": "gpt-5.6-luna"})
    s = ss.load_settings()                                   # what ui-config reads per request
    assert s["cascade_llm"] == "openai"
    assert ss.llm_model_for(s, s["cascade_llm"]) == "gpt-5.6-luna"
    assert json.loads(settings_file.read_text())["cascade_llm"] == "openai"
