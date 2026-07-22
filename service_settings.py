"""Persistent admin settings for which provider powers each service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = _ROOT / "admin_settings.json"

# LLM / pipeline switches admins can turn off to block accidental spend.
PROVIDER_IDS: tuple[str, ...] = ("deepseek", "glm", "groq", "openai", "realtime")
LLM_PROVIDER_IDS: tuple[str, ...] = ("deepseek", "glm", "groq", "openai")

DEFAULT_ENABLED_PROVIDERS: dict[str, bool] = {pid: True for pid in PROVIDER_IDS}

PROVIDER_LABELS: dict[str, str] = {
    "deepseek": "DeepSeek",
    "glm": "Zhipu GLM",
    "groq": "Groq",
    "openai": "OpenAI Chat (costly)",
    "realtime": "OpenAI Realtime (costly)",
}

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
    # When on, dump every LLM-bound message (incl. tool calls) under chats/messages/<session>/
    "save_llm_messages": False,
    # Silence after user speech before the turn is sent to the LLM (cascade voice).
    "vad_stop_secs": 0.2,
    # Flat booleans persist reliably from the admin UI (nested dicts were flaky in practice).
    "enable_deepseek": True,
    "enable_glm": True,
    "enable_groq": True,
    "enable_openai": True,
    "enable_realtime": True,
}

ALLOWED = {
    "stt": {"deepgram"},
    "tts": {"deepgram"},
    "cascade_llm": {"glm", "groq", "openai", "deepseek"},
    "cli_llm": {"groq", "glm", "openai", "deepseek"},
    "voice_pipeline_default": {"cascade", "realtime"},
}

_ENABLE_KEYS = {f"enable_{pid}" for pid in PROVIDER_IDS}
_BOOL_KEYS = {"debug_mode", "save_llm_messages"} | _ENABLE_KEYS


def enabled_providers_map(settings: dict[str, Any] | None = None) -> dict[str, bool]:
    s = settings or {}
    return {pid: bool(s.get(f"enable_{pid}", True)) for pid in PROVIDER_IDS}


def normalize_enabled_providers(raw: Any) -> dict[str, bool]:
    out = dict(DEFAULT_ENABLED_PROVIDERS)
    if isinstance(raw, dict):
        for key in PROVIDER_IDS:
            if key in raw:
                out[key] = bool(raw[key])
            elif f"enable_{key}" in raw:
                out[key] = bool(raw[f"enable_{key}"])
    return out


def apply_enabled_map(data: dict[str, Any], enabled: dict[str, bool]) -> dict[str, Any]:
    for pid in PROVIDER_IDS:
        data[f"enable_{pid}"] = bool(enabled.get(pid, True))
    return data


def is_provider_enabled(settings: dict[str, Any] | None, provider_id: str) -> bool:
    return bool(enabled_providers_map(settings).get(provider_id, True))


def first_enabled_llm(settings: dict[str, Any] | None = None) -> str:
    enabled = enabled_providers_map(settings)
    for pid in ("deepseek", "groq", "glm", "openai"):
        if enabled.get(pid, True):
            return pid
    raise ValueError("At least one LLM provider must stay enabled")


def resolve_llm_choice(settings: dict[str, Any], key: str) -> str:
    """Return a usable LLM id for cascade_llm / cli_llm (falls back if disabled)."""
    preferred = str(settings.get(key) or DEFAULT_SETTINGS.get(key) or "deepseek").strip().lower()
    if preferred in ALLOWED.get(key, set()) and is_provider_enabled(settings, preferred):
        return preferred
    return first_enabled_llm(settings)


def resolve_voice_pipeline(settings: dict[str, Any]) -> str:
    preferred = str(
        settings.get("voice_pipeline_default") or DEFAULT_SETTINGS["voice_pipeline_default"]
    ).strip().lower()
    if preferred == "realtime" and not is_provider_enabled(settings, "realtime"):
        return "cascade"
    if preferred in ALLOWED["voice_pipeline_default"]:
        return preferred
    return "cascade"


def assert_realtime_allowed(settings: dict[str, Any] | None = None) -> None:
    s = settings or load_settings()
    if not is_provider_enabled(s, "realtime"):
        raise RuntimeError(
            "OpenAI Realtime is disabled in Admin → Settings. "
            "Enable it under Cost controls, or use Cascade mode."
        )


def clamp_vad_stop_secs(value: Any) -> float:
    """Admin slider range 0.1–3.0s; default matches Pipecat stock VAD."""
    try:
        stop = float(value)
    except (TypeError, ValueError):
        stop = float(DEFAULT_SETTINGS["vad_stop_secs"])
    return max(0.1, min(3.0, round(stop, 2)))


def _coerce_settings(data: dict[str, Any]) -> dict[str, Any]:
    # Keep flat enable_* flags as source of truth; expose nested map for the UI/API.
    enabled = enabled_providers_map(data)
    apply_enabled_map(data, enabled)
    data["debug_mode"] = bool(data.get("debug_mode", False))
    data["save_llm_messages"] = bool(data.get("save_llm_messages", False))
    data["vad_stop_secs"] = clamp_vad_stop_secs(data.get("vad_stop_secs"))
    data["cascade_llm"] = resolve_llm_choice(data, "cascade_llm")
    data["cli_llm"] = resolve_llm_choice(data, "cli_llm")
    data["voice_pipeline_default"] = resolve_voice_pipeline(data)
    # Nested copy for clients (not written as a separate schema key in DEFAULT, but included in JSON).
    data["enabled_providers"] = enabled
    return data


def load_settings() -> dict[str, Any]:
    data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                # Legacy nested map from earlier builds.
                if isinstance(loaded.get("enabled_providers"), dict):
                    for pid, on in normalize_enabled_providers(loaded["enabled_providers"]).items():
                        data[f"enable_{pid}"] = on
                for key, value in loaded.items():
                    if key in DEFAULT_SETTINGS:
                        data[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    return _coerce_settings(data)


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    current = load_settings()

    # Accept either nested enabled_providers or flat enable_* flags.
    if "enabled_providers" in updates and updates["enabled_providers"] is not None:
        apply_enabled_map(current, normalize_enabled_providers(updates["enabled_providers"]))
        first_enabled_llm(current)

    for pid in PROVIDER_IDS:
        flat = f"enable_{pid}"
        if flat in updates and updates[flat] is not None:
            current[flat] = bool(updates[flat])
    if any(f"enable_{pid}" in updates for pid in LLM_PROVIDER_IDS):
        first_enabled_llm(current)

    for key, value in updates.items():
        if key not in DEFAULT_SETTINGS or key in _ENABLE_KEYS or key == "enabled_providers":
            continue
        if key in _BOOL_KEYS:
            current[key] = bool(value)
            continue
        if key == "vad_stop_secs":
            current[key] = clamp_vad_stop_secs(value)
            continue
        if key in ALLOWED and str(value) not in ALLOWED[key]:
            raise ValueError(f"Invalid value for {key}: {value}")
        current[key] = value

    current = _coerce_settings(current)
    # Persist flat flags + routing; also keep nested map for readability.
    to_write = {k: current[k] for k in DEFAULT_SETTINGS}
    to_write["enabled_providers"] = enabled_providers_map(current)
    SETTINGS_PATH.write_text(json.dumps(to_write, indent=2) + "\n", encoding="utf-8")
    apply_settings_to_env(current)
    return current


def apply_settings_to_env(settings: dict[str, Any] | None = None) -> None:
    """Push settings into process env so voice/CLI modules pick them up."""
    s = _coerce_settings(dict(settings or load_settings()))
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
    os.environ["LLM_PROVIDER"] = resolve_llm_choice(s, "cascade_llm")
    os.environ["CLI_LLM_PROVIDER"] = resolve_llm_choice(s, "cli_llm")
    os.environ["BOT_MODE"] = resolve_voice_pipeline(s)
    os.environ["ENABLED_REALTIME"] = "1" if is_provider_enabled(s, "realtime") else "0"
    os.environ["VAD_STOP_SECS"] = str(clamp_vad_stop_secs(s.get("vad_stop_secs")))


def options_catalog(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    s = settings or load_settings()
    enabled = enabled_providers_map(s)

    def llm_options(order: tuple[str, ...]) -> list[dict[str, Any]]:
        labels = {
            "deepseek": "DeepSeek",
            "glm": "Zhipu GLM",
            "groq": "Groq",
            "openai": "OpenAI Chat",
        }
        return [
            {
                "id": pid,
                "label": labels[pid] + ("" if enabled.get(pid, True) else " (disabled)"),
                "enabled": bool(enabled.get(pid, True)),
            }
            for pid in order
        ]

    return {
        "stt": [
            {"id": "deepgram", "label": "Deepgram STT", "enabled": True},
        ],
        "tts": [
            {"id": "deepgram", "label": "Deepgram Aura TTS", "enabled": True},
        ],
        "cascade_llm": llm_options(("deepseek", "glm", "groq", "openai")),
        "cli_llm": llm_options(("groq", "deepseek", "glm", "openai")),
        "voice_pipeline_default": [
            {"id": "cascade", "label": "Cascade (STT → LLM → TTS)", "enabled": True},
            {
                "id": "realtime",
                "label": "OpenAI Realtime"
                + ("" if enabled.get("realtime", True) else " (disabled)"),
                "enabled": bool(enabled.get("realtime", True)),
            },
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
        "enabled_providers": [
            {"id": pid, "label": PROVIDER_LABELS[pid], "enabled": bool(enabled.get(pid, True))}
            for pid in PROVIDER_IDS
        ],
    }
