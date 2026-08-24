"""retention.py purges only our hospital-ai traces from Langfuse, and never blocks local purges."""
from __future__ import annotations

import pytest

import retention


@pytest.fixture
def fake_api(monkeypatch):
    calls: list[tuple[str, str, dict | None, dict | None]] = []
    pages = {
        1: {"data": [{"id": "t1"}, {"id": "t2"}], "meta": {"totalPages": 2}},
        2: {"data": [{"id": "t3"}], "meta": {"totalPages": 2}},
    }

    def fake(method, path, *, params=None, body=None):
        calls.append((method, path, params, body))
        if method == "GET":
            return pages[int(params["page"])]
        return {}

    monkeypatch.setattr(retention, "_langfuse_request", fake)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    return calls


def test_purge_paginates_filters_by_tag_and_bulk_deletes(fake_api):
    n = retention.purge_langfuse_traces(1, dry_run=False)
    assert n == 3
    gets = [c for c in fake_api if c[0] == "GET"]
    assert [c[2]["page"] for c in gets] == [1, 2]
    assert all(c[2]["tags"] == "hospital-ai" for c in gets)          # never another project's traces
    assert all(c[2]["toTimestamp"].endswith("Z") for c in gets)
    deletes = [c for c in fake_api if c[0] == "DELETE"]
    assert deletes == [("DELETE", "/api/public/traces", None, {"traceIds": ["t1", "t2", "t3"]})]


def test_dry_run_counts_without_deleting(fake_api):
    assert retention.purge_langfuse_traces(1, dry_run=True) == 3
    assert not [c for c in fake_api if c[0] == "DELETE"]


def test_zero_days_or_no_keys_skips(monkeypatch, fake_api):
    assert retention.purge_langfuse_traces(0, dry_run=False) == 0
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY")
    assert retention.purge_langfuse_traces(1, dry_run=False) == 0
    assert fake_api == []


def test_policy_default_is_one_day(monkeypatch):
    monkeypatch.delenv("RETAIN_LANGFUSE_DAYS", raising=False)
    assert retention.policy()["langfuse"] == 1


def test_unreachable_langfuse_does_not_break_local_purges(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    for name in ("purge_sessions", "purge_tool_calls", "purge_timeline", "purge_llm_dumps",
                 "purge_metrics", "purge_backups"):
        monkeypatch.setattr(retention, name, lambda days, dry_run: 0)
    result = retention.run_retention(dry_run=True)
    assert result["langfuse"] == -1 and "langfuse_error" in result
