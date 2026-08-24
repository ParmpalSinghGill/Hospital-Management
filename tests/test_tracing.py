"""Langfuse tracing is opt-in, PHI-masked in HIPAA mode, and never breaks a turn."""
from __future__ import annotations

import pytest

import tracing


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_HOST",
        "LANGFUSE_TRACING_ENABLED",
        "LANGFUSE_MASK_PHI",
    ):
        monkeypatch.delenv(k, raising=False)
    tracing.reset_for_tests()
    yield
    tracing.reset_for_tests()


def _fake_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    # Unroutable port so nothing hangs; the SDK buffers and fails in background.
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")


def test_off_without_keys():
    assert not tracing.langfuse_enabled()
    assert tracing.get_handler() is None
    assert tracing.trace_config(session_id="s1") == {}
    cfg = {"configurable": {"thread_id": "t"}}
    assert tracing.with_tracing(cfg, session_id="s1") is cfg


def test_keys_but_paused(monkeypatch):
    _fake_keys(monkeypatch)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "0")
    assert tracing.langfuse_configured()
    assert not tracing.langfuse_enabled()
    assert tracing.trace_config(session_id="s1") == {}


def test_enabled_builds_handler_and_metadata(monkeypatch):
    pytest.importorskip("langfuse")
    _fake_keys(monkeypatch)
    monkeypatch.setenv("HIPAA_MODE", "0")
    cfg = tracing.with_tracing(
        {"configurable": {"thread_id": "thr-1"}, "callbacks": ["existing"]},
        session_id="call-9",
        user_id="usr-1",
        tags=["voice"],
        metadata={"thread_id": "thr-1"},
    )
    from langfuse.langchain import CallbackHandler

    assert cfg["configurable"] == {"thread_id": "thr-1"}
    assert cfg["callbacks"][0] == "existing"
    assert isinstance(cfg["callbacks"][1], CallbackHandler)
    md = cfg["metadata"]
    assert md["langfuse_session_id"] == "call-9"
    assert md["langfuse_user_id"] == "usr-1"
    assert md["langfuse_tags"] == ["voice"]
    assert md["langfuse_trace_name"] == tracing.DEFAULT_TRACE_NAME
    assert md["thread_id"] == "thr-1"
    # Handler is a process singleton.
    assert tracing.get_handler() is cfg["callbacks"][1]


def test_mask_follows_hipaa_mode(monkeypatch):
    monkeypatch.setenv("HIPAA_MODE", "1")
    assert tracing.mask_phi_enabled()
    monkeypatch.setenv("LANGFUSE_MASK_PHI", "0")
    assert not tracing.mask_phi_enabled()
    monkeypatch.delenv("LANGFUSE_MASK_PHI")
    monkeypatch.setenv("HIPAA_MODE", "0")
    assert not tracing.mask_phi_enabled()


def test_mask_redacts_identifiers():
    out = tracing.mask_for_trace(
        {"messages": [{"role": "user", "content": "call me on 9523641256 or a@b.com"}],
         "patient_name": "Amrit Kaur", "phone": "9523641256", "department": "Gynecology"}
    )
    blob = str(out)
    assert "9523641256" not in blob and "a@b.com" not in blob and "Amrit Kaur" not in blob
    assert out["department"] == "Gynecology"


def test_sdk_failure_degrades_to_no_tracing(monkeypatch):
    _fake_keys(monkeypatch)
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name.startswith("langfuse"):
            raise RuntimeError("sdk exploded")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert tracing.get_handler() is None
    assert tracing.trace_config(session_id="x") == {}
