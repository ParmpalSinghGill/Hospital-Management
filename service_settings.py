"""Persistent admin settings for which provider powers each service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = _ROOT / "admin_settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "stt": "deepgram",
    "tts": "deepgram",
    "cascade_llm": "deepseek",
    "cli_llm": "groq",
    "voice_pipeline_default": "cascade",
    "deepgram_voice": "aura-2-thalia-en",
    "glm_model": "glm-4-flash",
    "groq_model": "llama-3.3-70b-versatile",
    "deepseek_model": "deepseek-chat",
    "openai_realtime_model": "gpt-realtime",
    "openai_realtime_voice": "marin",
    "debug_mode": False,
}

ALLOWED = {
    "stt": {"deepgram"},
    "tts": {"deepgram"},
    "cascade_llm": {"glm", "groq", "openai", "deepseek"},
    "cli_llm": {"groq", "glm", "openai", "deepseek"},
    "voice_pipeline_default": {"cascade", "realtime"},
}


def load_settings() -> dict[str, Any]:
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update({k: v for k, v in loaded.items() if k in DEFAULT_SETTINGS})
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    current = load_settings()
    for key, value in updates.items():
        if key not in DEFAULT_SETTINGS:
            continue
        if key == "debug_mode":
            current[key] = bool(value)
            continue
        if key in ALLOWED and str(value) not in ALLOWED[key]:
            raise ValueError(f"Invalid value for {key}: {value}")
        current[key] = value
    SETTINGS_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    apply_settings_to_env(current)
    return current


def apply_settings_to_env(settings: dict[str, Any] | None = None) -> None:
    """Push settings into process env so voice/CLI modules pick them up."""
    s = settings or load_settings()
    os.environ["DEEPGRAM_VOICE"] = str(s.get("deepgram_voice") or DEFAULT_SETTINGS["deepgram_voice"])
    os.environ["GLM_MODEL"] = str(s.get("glm_model") or DEFAULT_SETTINGS["glm_model"])
    os.environ["GROQ_MODEL"] = str(s.get("groq_model") or DEFAULT_SETTINGS["groq_model"])
    os.environ["DEEPSEEK_MODEL"] = str(s.get("deepseek_model") or DEFAULT_SETTINGS["deepseek_model"])
    os.environ["OPENAI_REALTIME_MODEL"] = str(
        s.get("openai_realtime_model") or DEFAULT_SETTINGS["openai_realtime_model"]
    )
    os.environ["OPENAI_REALTIME_VOICE"] = str(
        s.get("openai_realtime_voice") or DEFAULT_SETTINGS["openai_realtime_voice"]
    )
    # Cascade graph LLM (voice)
    os.environ["LLM_PROVIDER"] = str(s.get("cascade_llm") or "deepseek")
    # Text CLI graph LLM
    os.environ["CLI_LLM_PROVIDER"] = str(s.get("cli_llm") or "groq")
    # Soft default voice mode (UI can still override per session)
    os.environ.setdefault("BOT_MODE", str(s.get("voice_pipeline_default") or "cascade"))


def options_catalog() -> dict[str, Any]:
    return {
        "stt": [
            {"id": "deepgram", "label": "Deepgram STT"},
        ],
        "tts": [
            {"id": "deepgram", "label": "Deepgram Aura TTS"},
        ],
        "cascade_llm": [
            {"id": "deepseek", "label": "DeepSeek"},
            {"id": "glm", "label": "Zhipu GLM"},
            {"id": "groq", "label": "Groq"},
            {"id": "openai", "label": "OpenAI Chat"},
        ],
        "cli_llm": [
            {"id": "groq", "label": "Groq"},
            {"id": "deepseek", "label": "DeepSeek"},
            {"id": "glm", "label": "Zhipu GLM"},
            {"id": "openai", "label": "OpenAI Chat"},
        ],
        "voice_pipeline_default": [
            {"id": "cascade", "label": "Cascade (STT → LLM → TTS)"},
            {"id": "realtime", "label": "OpenAI Realtime"},
        ],
        "openai_realtime_voices": [
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "sage",
            "shimmer",
            "verse",
            "marin",
            "cedar",
        ],
    }
