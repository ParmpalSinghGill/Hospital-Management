"""Tests for admin service settings and cost-control enable flags."""

from __future__ import annotations

import pytest

import service_settings as ss


@pytest.fixture()
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "admin_settings.json"
    monkeypatch.setattr(ss, "SETTINGS_PATH", path)
    return path


def test_defaults_enable_all_providers(settings_file):
    s = ss.load_settings()
    for pid in ("deepseek", "glm", "groq", "openai", "realtime"):
        assert s[f"enable_{pid}"] is True
        assert s["enabled_providers"][pid] is True


def test_disable_realtime_forces_cascade(settings_file):
    saved = ss.save_settings(
        {
            "voice_pipeline_default": "realtime",
            "enable_realtime": False,
            "enable_openai": True,
            "enable_deepseek": True,
            "enable_glm": True,
            "enable_groq": True,
        }
    )
    assert saved["voice_pipeline_default"] == "cascade"
    assert saved["enable_realtime"] is False
    assert ss.is_provider_enabled(saved, "realtime") is False
    with pytest.raises(RuntimeError, match="Realtime is disabled"):
        ss.assert_realtime_allowed(saved)


def test_disable_openai_via_nested_map_remaps_llm(settings_file):
    saved = ss.save_settings(
        {
            "cascade_llm": "openai",
            "cli_llm": "openai",
            "enabled_providers": {
                "deepseek": True,
                "glm": True,
                "groq": True,
                "openai": False,
                "realtime": True,
            },
        }
    )
    assert saved["enable_openai"] is False
    assert saved["cascade_llm"] != "openai"
    assert saved["cli_llm"] != "openai"


def test_flat_flags_persist_across_reload(settings_file):
    ss.save_settings(
        {
            "enable_openai": False,
            "enable_realtime": False,
            "enable_deepseek": True,
            "enable_glm": True,
            "enable_groq": True,
        }
    )
    raw = settings_file.read_text(encoding="utf-8")
    assert '"enable_openai": false' in raw
    assert '"enable_realtime": false' in raw
    again = ss.load_settings()
    assert again["enable_openai"] is False
    assert again["enable_realtime"] is False
    assert again["enabled_providers"]["openai"] is False


def test_cannot_disable_every_llm(settings_file):
    with pytest.raises(ValueError, match="At least one LLM"):
        ss.save_settings(
            {
                "enable_deepseek": False,
                "enable_glm": False,
                "enable_groq": False,
                "enable_openai": False,
            }
        )


def test_options_catalog_marks_disabled(settings_file):
    ss.save_settings({"enable_openai": False, "enable_realtime": False})
    catalog = ss.options_catalog()
    openai_opt = next(o for o in catalog["cascade_llm"] if o["id"] == "openai")
    assert openai_opt["enabled"] is False
    realtime_opt = next(o for o in catalog["voice_pipeline_default"] if o["id"] == "realtime")
    assert realtime_opt["enabled"] is False


def test_apply_settings_sets_enabled_realtime_env(settings_file, monkeypatch):
    monkeypatch.delenv("BOT_MODE", raising=False)
    monkeypatch.delenv("ENABLED_REALTIME", raising=False)
    ss.save_settings({"enable_realtime": False})
    assert ss.os.environ["ENABLED_REALTIME"] == "0"
    assert ss.os.environ["BOT_MODE"] == "cascade"


def test_save_llm_messages_setting_roundtrip(settings_file):
    saved = ss.save_settings({"save_llm_messages": True})
    assert saved["save_llm_messages"] is True
    assert ss.load_settings()["save_llm_messages"] is True
    saved_off = ss.save_settings({"save_llm_messages": False})
    assert saved_off["save_llm_messages"] is False


def test_vad_stop_secs_clamped_and_applied(settings_file, monkeypatch):
    monkeypatch.delenv("VAD_STOP_SECS", raising=False)
    assert ss.clamp_vad_stop_secs(0.01) == 0.1
    assert ss.clamp_vad_stop_secs(99) == 3.0
    saved = ss.save_settings({"vad_stop_secs": 1.25})
    assert saved["vad_stop_secs"] == 1.25
    assert ss.os.environ["VAD_STOP_SECS"] == "1.25"
    assert ss.load_settings()["vad_stop_secs"] == 1.25
