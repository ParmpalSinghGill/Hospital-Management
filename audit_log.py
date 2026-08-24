"""Tamper-evident audit trail for PHI access (HIPAA §164.312(b)).

One JSON line per event in ``<audit_dir>/audit.jsonl``; each line carries the
SHA-256 of the previous line (``prev``) and of itself (``hash``) so deletion or
edits break the chain. ``verify_chain()`` checks it. The audit log itself
never stores raw identifiers — subjects are ``patient_id`` / hashed phone.

Typical events: ``admin.login``, ``admin.patient.read``, ``tool.lookup_patient``,
``tool.book_appointment``, ``chat.delete``, ``settings.update``,
``retention.purge``, ``backup.create``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import security_config as sc

_LOCK = threading.RLock()
_LAST_HASH: str | None = None
_GENESIS = "0" * 64


def audit_path() -> Path:
    return sc.audit_dir() / "audit.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def subject_ref(*, patient_id: str | None = None, phone: str | None = None) -> dict[str, str]:
    """Reference a patient without storing raw identifiers."""
    out: dict[str, str] = {}
    if patient_id:
        out["patient_id"] = str(patient_id)
    if phone:
        digits = "".join(ch for ch in str(phone) if ch.isdigit())
        if digits:
            out["phone_hash"] = hashlib.sha256(
                (sc.server_secret() + ":" + digits[-10:]).encode()
            ).hexdigest()[:16]
    return out


def _tail_hash(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return _GENESIS
            step = min(size, 4096)
            fh.seek(size - step)
            chunk = fh.read(step)
        last = chunk.rstrip(b"\n").split(b"\n")[-1]
        rec = json.loads(last.decode("utf-8"))
        return str(rec.get("hash") or _GENESIS)
    except (OSError, ValueError):
        return _GENESIS


def _line_hash(prev: str, body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()


def audit(
    event: str,
    *,
    actor: str = "system",
    channel: str = "",
    session_id: str = "",
    subject: dict[str, Any] | None = None,
    outcome: str = "ok",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append one audit record. Never raises (audit must not break the app)."""
    global _LAST_HASH
    try:
        body: dict[str, Any] = {
            "at": _now(),
            "event": str(event),
            "actor": str(actor or "system")[:120],
            "outcome": str(outcome or "ok")[:40],
        }
        if channel:
            body["channel"] = str(channel)[:40]
        if session_id:
            body["session_id"] = str(session_id)[:120]
        if subject:
            body["subject"] = {k: str(v)[:120] for k, v in subject.items() if v}
        if detail and isinstance(detail, dict):
            # keep detail small and non-identifying (callers pass ids/counts only)
            body["detail"] = {
                str(k)[:60]: (v if isinstance(v, (int, float, bool)) else str(v)[:200])
                for k, v in detail.items()
            }
        path = audit_path()
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.parent.chmod(0o700)
            except OSError:
                pass
            # Several processes (web app, MCP server, bots) append to the same file:
            # take an exclusive file lock, re-read the tail hash, then append — never
            # trust an in-memory "last hash".
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                prev = _tail_hash(path)
                body["prev"] = prev
                body["hash"] = _line_hash(prev, body)
                line = json.dumps(body, sort_keys=True, ensure_ascii=False)
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            _LAST_HASH = body["hash"]
        return body
    except Exception:  # noqa: BLE001
        return None


def iter_records(path: Path | None = None) -> Iterator[dict[str, Any]]:
    p = path or audit_path()
    if not p.exists():
        return iter(())

    def _gen() -> Iterator[dict[str, Any]]:
        with p.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except ValueError:
                    yield {"_corrupt": raw[:200]}

    return _gen()


def verify_chain(path: Path | None = None) -> dict[str, Any]:
    """Walk the file and confirm every hash links to the previous one."""
    prev = _GENESIS
    count = 0
    for rec in iter_records(path):
        if "_corrupt" in rec:
            return {"ok": False, "records": count, "error": "corrupt line", "at": count + 1}
        expected_prev = rec.get("prev")
        body = {k: v for k, v in rec.items() if k != "hash"}
        if expected_prev != prev:
            return {"ok": False, "records": count, "error": "broken link", "at": count + 1}
        if _line_hash(prev, body) != rec.get("hash"):
            return {"ok": False, "records": count, "error": "hash mismatch", "at": count + 1}
        prev = str(rec.get("hash"))
        count += 1
    return {"ok": True, "records": count, "head": prev}


def recent(limit: int = 100, *, event_prefix: str = "") -> list[dict[str, Any]]:
    rows = [r for r in iter_records() if not event_prefix or str(r.get("event", "")).startswith(event_prefix)]
    return rows[-max(1, min(limit, 1000)):][::-1]
