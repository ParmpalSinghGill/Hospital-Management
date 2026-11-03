"""What voice speculation costs (extra LLM calls) and what it saves (time).

One row per guess goes to ``metrics/speculation.csv``. Running totals for the
call are kept on the call log JSON under ``meta.speculation``.

Time saved on a hit: without speculation the run would start at commit and take
``D``; with it, the run started at the pause, so the reply is ready at
``max(done, commit)``. Saved = ``min(D, commit − start)`` — all of ``D`` when the
guess finished before the turn committed, otherwise the head start.

A miss saves nothing, and every LLM call it made is an extra call.

    python speculation_stats.py [session_id]   # summary from the CSV
"""

from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_CSV_PATH = Path(__file__).parent / "metrics" / "speculation.csv"
SPECULATION_CSV = Path(os.environ.get("SPECULATION_CSV", DEFAULT_CSV_PATH))

CSV_HEADER = [
    "session_id",
    "started_at",
    "outcome",  # hit | miss | failed
    "reason",  # why a miss was dropped (text_changed, resumed, text_mismatch, ...)
    "text",
    "guess_ms",  # how long the guessed run took
    "head_start_ms",  # pause → commit (hits only)
    "saved_ms",  # hits only; 0 otherwise
    "wait_after_commit_ms",  # hits only: guess still running when turn committed
    "llm_calls",
    "llm_ms",
]

_LOCK = threading.Lock()
_TOTALS: dict[str, dict[str, Any]] = {}


def _empty_totals() -> dict[str, Any]:
    return {
        "guesses": 0,
        "hits": 0,
        "misses": 0,
        "failed": 0,
        "hit_rate": 0.0,
        "extra_llm_calls": 0,  # made by guesses that were thrown away
        "extra_llm_ms": 0.0,
        "hit_llm_calls": 0,
        "saved_ms_total": 0.0,
        "saved_ms_avg_per_hit": 0.0,
        "miss_reasons": {},
    }


def _add(totals: dict[str, Any], row: dict[str, Any]) -> None:
    outcome = row["outcome"]
    calls = int(row.get("llm_calls") or 0)
    llm_ms = float(row.get("llm_ms") or 0.0)
    totals["guesses"] += 1
    if outcome == "hit":
        totals["hits"] += 1
        totals["hit_llm_calls"] += calls
        totals["saved_ms_total"] = round(totals["saved_ms_total"] + float(row.get("saved_ms") or 0), 1)
    else:
        totals["misses" if outcome == "miss" else "failed"] += 1
        totals["extra_llm_calls"] += calls
        totals["extra_llm_ms"] = round(totals["extra_llm_ms"] + llm_ms, 1)
        reason = row.get("reason") or outcome
        totals["miss_reasons"][reason] = totals["miss_reasons"].get(reason, 0) + 1
    totals["hit_rate"] = round(totals["hits"] / totals["guesses"], 3)
    if totals["hits"]:
        totals["saved_ms_avg_per_hit"] = round(totals["saved_ms_total"] / totals["hits"], 1)


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


def record_guess(
    session_id: str,
    *,
    outcome: str,
    text: str,
    started_wall: datetime,
    started_at: float,
    done_at: float | None,
    committed_at: float | None = None,
    reason: str = "",
    llm_calls: int = 0,
    llm_ms: float = 0.0,
) -> dict[str, Any]:
    """Log one finished guess. Times are ``time.monotonic()`` seconds."""
    guess_s = None if done_at is None else max(0.0, done_at - started_at)
    head_s = saved_s = wait_s = None
    if outcome == "hit" and committed_at is not None:
        head_s = max(0.0, committed_at - started_at)
        saved_s = head_s if guess_s is None else min(guess_s, head_s)
        if done_at is not None:
            wait_s = max(0.0, done_at - committed_at)
    row = {
        "session_id": session_id,
        "started_at": started_wall.isoformat(timespec="milliseconds"),
        "outcome": outcome,
        "reason": reason,
        "text": (text or "")[:200],
        "guess_ms": _ms(guess_s) or 0.0,
        "head_start_ms": _ms(head_s) or "",
        "saved_ms": _ms(saved_s) or 0.0,
        "wait_after_commit_ms": "" if wait_s is None else _ms(wait_s),
        "llm_calls": int(llm_calls),
        "llm_ms": round(float(llm_ms), 1),
    }
    with _LOCK:
        totals = _TOTALS.setdefault(session_id, _empty_totals())
        _add(totals, row)
        snapshot = {**totals, "miss_reasons": dict(totals["miss_reasons"])}
        try:
            SPECULATION_CSV.parent.mkdir(parents=True, exist_ok=True)
            new = not SPECULATION_CSV.exists() or SPECULATION_CSV.stat().st_size == 0
            with SPECULATION_CSV.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_HEADER)
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError:
            pass
    _save_to_call_log(session_id, snapshot)
    logger.info(
        f"speculation {outcome}{f' ({reason})' if reason else ''}: "
        f"saved={row['saved_ms']}ms llm_calls={row['llm_calls']} | call totals: "
        f"{snapshot['hits']}/{snapshot['guesses']} hits, "
        f"extra_llm_calls={snapshot['extra_llm_calls']}, saved={snapshot['saved_ms_total']}ms"
    )
    return row


def session_totals(session_id: str) -> dict[str, Any]:
    with _LOCK:
        t = _TOTALS.get(session_id)
        return {**t, "miss_reasons": dict(t["miss_reasons"])} if t else _empty_totals()


def _save_to_call_log(session_id: str, totals: dict[str, Any]) -> None:
    if not session_id:
        return
    try:
        import chat_log_core as core

        with core._LOCK:
            path = core._path_for(session_id)
            data = core._read(path)
            if not data:
                return
            data.setdefault("meta", {})["speculation"] = totals
            data["updated_at"] = core._utc_iso()
            core._write(path, data)
    except Exception as e:  # noqa: BLE001 - stats must never break a call
        logger.debug(f"speculation stats not saved to call log: {e}")


def summarize_csv(session_id: str = "", path: Path | None = None) -> dict[str, Any]:
    """Totals from the CSV — for one call, or all calls when no id is given."""
    totals = _empty_totals()
    p = path or SPECULATION_CSV
    if not p.exists():
        return totals
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if session_id and row.get("session_id") != session_id:
                continue
            _add(totals, row)
    return totals


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    import json

    sid = sys.argv[1] if len(sys.argv) > 1 else ""
    print(json.dumps(summarize_csv(sid), indent=2))
