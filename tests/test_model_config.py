"""Model.py builds provider clients with the right per-model quirks."""
from __future__ import annotations

import pytest

import Model


def _payload(llm, **kwargs):
    """Request body ChatOpenAI would send (no network)."""
    return llm._get_request_payload([("human", "hi")], **kwargs)


def test_openai_default_is_luna_and_omits_temperature(monkeypatch):
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = Model._init_openai_model(None)
    body = _payload(llm)
    assert body["model"] == "gpt-5.6-luna"
    assert "temperature" not in body
    assert body["reasoning_effort"] == "none"
    assert body["store"] is False


def test_openai_legacy_model_keeps_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    body = _payload(Model._init_openai_model(None))
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0
    assert "reasoning_effort" not in body


def test_deepseek_default_is_v4_flash_with_thinking_disabled(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    llm = Model._init_deepseek_model(None)
    body = _payload(llm)
    assert body["model"] == "deepseek-v4-flash"
    assert body["extra_body"] == {"thinking": {"type": "disabled"}}
    assert body["parallel_tool_calls"] is False
    assert llm.openai_api_base == "https://api.deepseek.com"


def test_deepseek_thinking_can_be_reenabled(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_THINKING", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert "extra_body" not in _payload(Model._init_deepseek_model(None))


def test_deepseek_legacy_model_has_no_thinking_flag(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    assert "extra_body" not in _payload(Model._init_deepseek_model(None))


@pytest.mark.parametrize(
    "provider, env, expected_model",
    [("openai", "OPENAI_API_KEY", "gpt-5.6-luna"), ("deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash")],
)
def test_init_model_dispatch(monkeypatch, provider, env, expected_model):
    monkeypatch.setenv(env, "sk-test")
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("HIPAA_MODE", "0")  # dev posture: all providers allowed
    llm = Model._init_model(callbacks=None, provider=provider)
    assert llm.model_name == expected_model
