"""Retention & disposal (HIPAA §164.310(d)(2)) — scheduled purge of PHI stores.

Policy (days, env-overridable):
  RETAIN_SESSIONS_DAYS   = 90    chats/sessions/*.json, chats/chat-*.jsonl
  RETAIN_TOOL_CALLS_DAYS = 30    entries inside chats/tool_call.json
  RETAIN_TIMELINE_DAYS   = 30    chats/timeline.jsonl lines
  RETAIN_LLM_DUMPS_DAYS  = 7     chats/messages/<session>/
  RETAIN_METRICS_DAYS    = 90    metrics/bot_messages.csv rows
  RETAIN_BACKUPS_DAYS    = 35    backups/*
  RETAIN_AUDIT_DAYS      = 2190  audit/audit.jsonl is NOT purged (6 years) — rotate off-host instead
  RETAIN_LANGFUSE_DAYS   = 1     Langfuse traces tagged ``hospital-ai`` (self-hosted OSS has no
                                 built-in retention; purged via the public API when LANGFUSE_* set)

Run: ``python retention.py [--dry-run]`` or call ``start_scheduler()`` at app start.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import security_config as sc


def _days(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def policy() -> dict[str, int]:
    return {
        "sessions": _days("RETAIN_SESSIONS_DAYS", 90),
        "tool_calls": _days("RETAIN_TOOL_CALLS_DAYS", 30),
        "timeline": _days("RETAIN_TIMELINE_DAYS", 30),
        "llm_dumps": _days("RETAIN_LLM_DUMPS_DAYS", 7),
        "metrics": _days("RETAIN_METRICS_DAYS", 90),
        "backups": _days("RETAIN_BACKUPS_DAYS", 35),
        "audit": _days("RETAIN_AUDIT_DAYS", 2190),
        "langfuse": _days("RETAIN_LANGFUSE_DAYS", 1),
    }


def _cutoff(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _secure_delete(path: Path, dry_run: bool) -> None:
    """Best-effort overwrite + unlink (helps on non-CoW filesystems)."""
    if dry_run:
        return
    try:
        size = path.stat().st_size
        if size and size < 64 * 1024 * 1024:
            with path.open("r+b") as fh:
                fh.write(b"\0" * size)
                fh.flush()
                os.fsync(fh.fileno())
    except OSError:
        pass
    try:
        path.unlink()
    except OSError:
        pass


def _rmtree_secure(folder: Path, dry_run: bool) -> None:
    for p in sorted(folder.rglob("*"), reverse=True):
        if p.is_file():
            _secure_delete(p, dry_run)
        elif p.is_dir() and not dry_run:
            try:
                p.rmdir()
            except OSError:
                pass
    if not dry_run:
        try:
            folder.rmdir()
        except OSError:
            pass


def purge_sessions(days: int, dry_run: bool) -> int:
    chats = sc.chats_dir()
    n = 0
    cut = _cutoff(days)
    for pattern in ("sessions/sess-*.json", "sessions/call-*.json", "chat-*.jsonl"):
        for p in chats.glob(pattern):
            try:
                if _mtime(p) < cut:
                    _secure_delete(p, dry_run)
                    n += 1
            except OSError:
                continue
    return n


def purge_tool_calls(days: int, dry_run: bool) -> int:
    path = sc.chats_dir() / "tool_call.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    calls = data.get("tool_calls") if isinstance(data, dict) else None
    if not isinstance(calls, list):
        return 0
    cut = _cutoff(days)
    keep = [c for c in calls if (_parse_iso((c or {}).get("at")) or datetime.now(timezone.utc)) >= cut]
    removed = len(calls) - len(keep)
    if removed and not dry_run:
        data["tool_calls"] = keep
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return removed


def purge_timeline(days: int, dry_run: bool) -> int:
    path = sc.chats_dir() / "timeline.jsonl"
    if not path.exists():
        return 0
    cut = _cutoff(days)
    kept: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            try:
                at = _parse_iso(json.loads(raw).get("at"))
            except ValueError:
                at = None
            if at and at < cut:
                removed += 1
            else:
                kept.append(raw)
    if removed and not dry_run:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        os.replace(tmp, path)
    return removed


def purge_llm_dumps(days: int, dry_run: bool) -> int:
    root = sc.chats_dir() / "messages"
    if not root.exists():
        return 0
    cut = _cutoff(days)
    n = 0
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        try:
            newest = max((_mtime(p) for p in folder.rglob("*") if p.is_file()), default=_mtime(folder))
        except OSError:
            continue
        if newest < cut:
            _rmtree_secure(folder, dry_run)
            n += 1
    return n


def purge_metrics(days: int, dry_run: bool) -> int:
    path = sc.metrics_dir() / "bot_messages.csv"
    if not path.exists():
        return 0
    cut_date = _cutoff(days).date()
    removed = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    try:
        di = header.index("user_message_date")
    except ValueError:
        return 0
    keep = []
    for r in body:
        try:
            d = datetime.strptime(r[di], "%Y-%m-%d").date() if len(r) > di and r[di] else None
        except ValueError:
            d = None
        if d and d < cut_date:
            removed += 1
        else:
            keep.append(r)
    if removed and not dry_run:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(keep)
        os.replace(tmp, path)
    return removed


def purge_backups(days: int, dry_run: bool) -> int:
    root = sc.backups_dir()
    if not root.exists():
        return 0
    cut = _cutoff(days)
    n = 0
    for p in root.iterdir():
        if p.is_file() and _mtime(p) < cut:
            _secure_delete(p, dry_run)
            n += 1
    return n


# ---------------------------------------------------------------- Langfuse ----
LANGFUSE_TRACE_TAG = "hospital-ai"   # only our traces — the project may be shared
_LANGFUSE_PAGE = 100


def _langfuse_request(method: str, path: str, *, params: dict[str, Any] | None = None,
                      body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Minimal authenticated call to the Langfuse public API (stdlib only)."""
    import base64
    import urllib.parse
    import urllib.request

    base = (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").rstrip("/")
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode())
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — operator-configured URL
        raw = resp.read()
    return json.loads(raw) if raw else {}


def purge_langfuse_traces(days: int, dry_run: bool) -> int:
    """Delete Langfuse traces tagged ``hospital-ai`` older than *days* (0 = skip)."""
    if days <= 0:
        return 0
    try:
        import tracing
    except Exception:  # noqa: BLE001
        return 0
    if not tracing.langfuse_configured():
        return 0
    cutoff = _cutoff(days).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    ids: list[str] = []
    page = 1
    while True:
        data = _langfuse_request(
            "GET", "/api/public/traces",
            params={"tags": LANGFUSE_TRACE_TAG, "toTimestamp": cutoff, "limit": _LANGFUSE_PAGE, "page": page},
        )
        rows = data.get("data") or []
        ids.extend(str(r.get("id")) for r in rows if r.get("id"))
        meta = data.get("meta") or {}
        if not rows or page >= int(meta.get("totalPages") or 1):
            break
        page += 1
    if dry_run or not ids:
        return len(ids)
    for i in range(0, len(ids), _LANGFUSE_PAGE):
        _langfuse_request("DELETE", "/api/public/traces", body={"traceIds": ids[i:i + _LANGFUSE_PAGE]})
    return len(ids)


def run_retention(*, dry_run: bool = False) -> dict[str, int]:
    pol = policy()
    result = {
        "sessions": purge_sessions(pol["sessions"], dry_run),
        "tool_calls": purge_tool_calls(pol["tool_calls"], dry_run),
        "timeline": purge_timeline(pol["timeline"], dry_run),
        "llm_dumps": purge_llm_dumps(pol["llm_dumps"], dry_run),
        "metrics": purge_metrics(pol["metrics"], dry_run),
        "backups": purge_backups(pol["backups"], dry_run),
    }
    try:
        result["langfuse"] = purge_langfuse_traces(pol["langfuse"], dry_run)
    except Exception as exc:  # noqa: BLE001 — an unreachable Langfuse must not stop local purges
        result["langfuse"] = -1
        result["langfuse_error"] = str(exc)[:200]  # type: ignore[assignment]
    try:
        from audit_log import audit

        audit("retention.purge", actor="scheduler", outcome="dry_run" if dry_run else "ok", detail=result)
    except Exception:  # noqa: BLE001
        pass
    return result


_THREAD: threading.Thread | None = None


def start_scheduler(interval_secs: int = 24 * 3600, *, initial_delay_secs: int = 60) -> None:
    """Run ``run_retention`` daily in a daemon thread (idempotent)."""
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return

    def _loop() -> None:
        time.sleep(initial_delay_secs)
        while True:
            try:
                run_retention()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval_secs)

    _THREAD = threading.Thread(target=_loop, name="retention-scheduler", daemon=True)
    _THREAD.start()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Purge PHI stores per retention policy.")
    parser.add_argument("--dry-run", action="store_true", help="report only, delete nothing")
    args = parser.parse_args(argv)
    print("policy (days):", json.dumps(policy()))
    print(("would remove" if args.dry_run else "removed") + ":", json.dumps(run_retention(dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
