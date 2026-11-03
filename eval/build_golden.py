"""Build a golden turn-level dataset from real call logs.

Each item is a snapshot of the conversation *just before* one real caller turn:
the history the agent saw (user / assistant text plus earlier tool calls and
their results), the caller's next words, and what production did next (which
agent answered, which tool it called first). ``run_golden_latency.py`` replays
each snapshot as one LLM call to measure latency and check the first action.

Sources (read-only):
  chats/sessions/*.json   one file per call: turns, agent, reply text, timestamps
  chats/tool_call.json    every tool call with session_id, time, args, result

Seeded demo sessions (pipeline_mode "seed") and test sessions are skipped.

    python eval/build_golden.py                 # ~50 items -> eval/golden/golden_turns.jsonl
    python eval/build_golden.py --size 60 --seed 7

Labels: ``expected`` starts as what production did (``review: "pending"``).
Production is not always right, so read the items and fix ``expected`` where
needed, then set ``review`` to ``"ok"``. Re-running the builder keeps reviewed
labels for items it selects again (matched by ``id``).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reception_routing import (  # noqa: E402  (pure functions, no I/O)
    _is_greeting_only,
    _is_short_affirmation,
    _sticky_route_from_history,
)

SESSIONS_DIR = ROOT / "chats" / "sessions"
TOOL_CALLS = ROOT / "chats" / "tool_call.json"
DEFAULT_OUT = ROOT / "eval" / "golden" / "golden_turns.jsonl"

SPECIALISTS = ("booking", "cancelling", "rescheduling", "prescriptions")
SKIP_ID_PREFIXES = ("timing", "test")
_OTHER_INDUSTRY_RE = re.compile(r"\b(tables?|rooms?|nights|reservations?)\b", re.IGNORECASE)

# Clinic rules (db_core.py): open 09:00–17:00, lunch 14:00–15:00, 10-minute slots.
CLINIC_OPEN, CLINIC_CLOSE = 9 * 60, 17 * 60
LUNCH_START, LUNCH_END = 14 * 60, 15 * 60
SLOT_MINUTES = 10

# Category quotas as shares of --size, in pick order (scarcest first).
QUOTAS = {
    "slot_edge": 0.24,
    "ambiguous_routing": 0.24,
    "sticky_short": 0.24,
    "tool_turn": 0.16,
    "other": 0.12,
}
MAX_PER_SESSION_PER_CATEGORY = 3

_SWITCH_RE = re.compile(
    r"\b(actually|instead|rather|change|move|cancel|reschedule|another|different|wait|no,)\b",
    re.IGNORECASE,
)
# A clock time: "3pm", "3:15", "at 3", "10.30 am". Bare numbers (phone digits) don't count.
_CLOCK_RE = re.compile(
    r"(?:\bat\s+(?P<h1>\d{1,2})(?:[:.](?P<m1>\d{2}))?\s*(?P<ap1>[ap]\.?\s?m\.?)?)"
    r"|(?:\b(?P<h2>\d{1,2})[:.](?P<m2>\d{2})\s*(?P<ap2>[ap]\.?\s?m\.?)?)"
    r"|(?:\b(?P<h3>\d{1,2})\s*(?P<ap3>[ap]\.?\s?m\.?)(?![a-z]))",
    re.IGNORECASE,
)
_SOFT_TIME_RE = re.compile(
    r"\b(tomorrow|today|tonight|morning|afternoon|evening|noon|lunch|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|weekend|next week)\b",
    re.IGNORECASE,
)
_OUTSIDE_WORDS_RE = re.compile(r"\b(tonight|night|evening|early morning|weekend|saturday|sunday)\b", re.I)
_EXISTING_BOOKING_RE = re.compile(
    r"already (?:have|has|got) (?:an? )?(?:active |upcoming |existing )?appointment|already booked"
    r"|existing appointment|active appointment",
    re.IGNORECASE,
)
_SLOT_REFUSAL_RE = re.compile(
    r"not available|isn't available|unavailable|no (?:open |free )?slots?|fully booked|closed"
    r"|outside (?:our |clinic |the clinic'?s? )?hours|lunch|only (?:take|book) .*(?:minute|10)",
    re.IGNORECASE,
)


# ── loading ────────────────────────────────────────────────────────────────


def _ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _other_industry(session: dict[str, Any]) -> bool:
    """Restaurant / hotel test calls: replaying them on the clinic prompts is meaningless."""
    text = " ".join(
        f"{(t.get('payload') or {}).get('user_input_text', '')} {(t.get('payload') or {}).get('bot_response_text', '')}"
        for t in session.get("interactions") or []
    )
    return bool(_OTHER_INDUSTRY_RE.search(text))


def load_sessions(sessions_dir: Path = SESSIONS_DIR) -> list[dict[str, Any]]:
    """Real sessions only, oldest first."""
    out = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = str(data.get("session_id") or "")
        if data.get("pipeline_mode") == "seed" or not sid or sid.startswith(SKIP_ID_PREFIXES):
            continue
        if not data.get("interactions") or _other_industry(data):
            continue
        out.append(data)
    out.sort(key=lambda d: str(d.get("session_start_time") or ""))
    return out


def load_tool_calls(path: Path = TOOL_CALLS) -> dict[str, list[dict[str, Any]]]:
    """Tool calls grouped by session, oldest first."""
    if not path.exists():
        return {}
    store = json.loads(path.read_text(encoding="utf-8"))
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in store.get("tool_calls", []):
        sid = entry.get("session_id")
        if sid and entry.get("tool"):
            by_session[sid].append(entry)
    for calls in by_session.values():
        calls.sort(key=lambda e: _ts(e.get("at")) or 0.0)
    return by_session


def tools_per_turn(turns: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[list[dict]]:
    """Assign each tool call to the turn whose window [sent_i, sent_i+1) holds it."""
    starts = [_ts((t.get("timestamps") or {}).get("user_sent_time")) for t in turns]
    out: list[list[dict]] = [[] for _ in turns]
    for call in calls:
        at = _ts(call.get("at"))
        if at is None:
            continue
        idx = None
        for i, start in enumerate(starts):
            if start is not None and start <= at:
                idx = i
        if idx is not None:
            out[idx].append(call)
    return out


# ── snapshot building ──────────────────────────────────────────────────────


def _tool_exchange(call: dict[str, Any], n: int, turn_no: int) -> list[dict[str, Any]]:
    """An assistant tool request and its result, as production stored them in state."""
    call_id = str(call.get("tool_call_id") or f"golden_t{turn_no}_{n}")
    result = call.get("result")
    if result is None:
        result = {"ok": False, "message": call.get("error") or "tool failed"}
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "name": call["tool"], "args": call.get("arguments") or {}}],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": call["tool"],
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


def _routing_view(history: list[dict[str, Any]], user_text: str) -> list[tuple[str, str]]:
    msgs = [("human" if m["role"] == "user" else "ai", m.get("content") or "")
            for m in history if m["role"] in ("user", "assistant") and m.get("content")]
    msgs.append(("human", user_text))
    return msgs


def _clock_minutes(text: str) -> list[tuple[int, int]]:
    """(hour, minute) for each clock time in ``text``, 24h, best guess for am/pm."""
    found = []
    for m in _CLOCK_RE.finditer(text or ""):
        h = m.group("h1") or m.group("h2") or m.group("h3")
        mi = m.group("m1") or m.group("m2") or "0"
        ap = (m.group("ap1") or m.group("ap2") or m.group("ap3") or "").lower().replace(".", "").replace(" ", "")
        hour, minute = int(h), int(mi)
        if hour > 23 or minute > 59:
            continue
        if ap.startswith("p") and hour < 12:
            hour += 12
        elif ap.startswith("a") and hour == 12:
            hour = 0
        elif not ap and 1 <= hour <= 7:
            hour += 12  # "at 3" in a clinic means 15:00
        found.append((hour, minute))
    return found


def tag_turn(
    *,
    user_text: str,
    bot_text: str,
    prev_agent: str,
    agent: str,
    rule_route: str | None,
    tools: list[dict[str, Any]],
    turn_index: int,
) -> tuple[set[str], list[str]]:
    """Categories this turn fits, plus finer reasons (shown to the reviewer)."""
    cats: set[str] = set()
    why: list[str] = []
    low = user_text.lower()
    words = user_text.split()

    # Slot edge cases: lunch, outside hours, off-grid minutes, existing booking, refusals.
    for hour, minute in _clock_minutes(user_text):
        t = hour * 60 + minute
        if LUNCH_START <= t < LUNCH_END:
            why.append(f"lunch clash {hour:02d}:{minute:02d}")
        if t < CLINIC_OPEN or t >= CLINIC_CLOSE:
            why.append(f"outside hours {hour:02d}:{minute:02d}")
        if minute % SLOT_MINUTES:
            why.append(f"off-grid minute {hour:02d}:{minute:02d}")
    if "lunch" in low:
        why.append("mentions lunch")
    if _OUTSIDE_WORDS_RE.search(user_text):
        why.append("outside-hours wording")
    if _EXISTING_BOOKING_RE.search(bot_text):
        why.append("existing active booking")
    if _SLOT_REFUSAL_RE.search(bot_text) and (_clock_minutes(user_text) or _SOFT_TIME_RE.search(user_text)):
        why.append("slot refused")
    if any(t["tool"] in ("book_appointment", "reschedule_appointment") and t.get("ok") is False for t in tools):
        why.append("booking tool refused")
    if why:
        cats.add("slot_edge")

    # Ambiguous routing: intent switches, router fallback, rules disagreeing with what ran.
    amb = []
    if prev_agent and _SWITCH_RE.search(user_text) and not _is_greeting_only(user_text):
        amb.append("switch wording")
    if prev_agent and agent and agent != prev_agent and not _is_greeting_only(user_text):
        amb.append(f"agent changed {prev_agent}->{agent}")
    if rule_route is None and turn_index > 0:
        amb.append("rules undecided (router LLM)")
    if rule_route and agent and rule_route != agent:
        amb.append(f"rules say {rule_route}, ran {agent}")
    if amb:
        cats.add("ambiguous_routing")
        why += amb

    # Short sticky replies mid-flow: "yes", digits, "tomorrow morning", a name.
    if prev_agent in SPECIALISTS and len(words) <= 6:
        digits = sum(ch.isdigit() for ch in user_text)
        kind = (
            "affirmation" if _is_short_affirmation(user_text)
            else "digits" if digits >= 4
            else "time" if (_clock_minutes(user_text) or _SOFT_TIME_RE.search(user_text))
            else "short answer"
        )
        cats.add("sticky_short")
        why.append(f"sticky {kind} after {prev_agent}")

    if tools:
        cats.add("tool_turn")
        why.append("tools: " + ",".join(t["tool"] for t in tools))

    if not cats:
        cats.add("other")
        why.append("greeting" if _is_greeting_only(user_text) else "general")
    return cats, why


def snapshots_for_session(session: dict[str, Any], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns = sorted(session.get("interactions") or [], key=lambda t: t.get("turn_number") or 0)
    per_turn_tools = tools_per_turn(turns, calls)
    sid = session["session_id"]
    history: list[dict[str, Any]] = []
    prev_agent = ""
    out = []
    for i, turn in enumerate(turns):
        payload = turn.get("payload") or {}
        user_text = str(payload.get("user_input_text") or "").strip()
        bot_text = str(payload.get("bot_response_text") or "").strip()
        agent = str(turn.get("agent") or "").strip()
        tools = per_turn_tools[i]

        if user_text:
            rule_route = _sticky_route_from_history(_routing_view(history, user_text), last_agent=prev_agent)
            cats, why = tag_turn(
                user_text=user_text, bot_text=bot_text, prev_agent=prev_agent, agent=agent,
                rule_route=rule_route, tools=tools, turn_index=i,
            )
            server = turn.get("server") or {}
            out.append({
                "id": f"{sid}#t{turn.get('turn_number')}",
                "categories": sorted(cats),
                "why": why,
                "source": {
                    "session_id": sid,
                    "turn_number": turn.get("turn_number"),
                    "pipeline_mode": session.get("pipeline_mode") or "",
                    "user_sent_time": (turn.get("timestamps") or {}).get("user_sent_time"),
                },
                "history": json.loads(json.dumps(history)),  # copy
                "user_text": user_text,
                "prev_agent": prev_agent,
                "rule_route": rule_route,
                "logged": {
                    "agent": agent,
                    "reply": bot_text,
                    "tools": [
                        {"name": t["tool"], "args": t.get("arguments") or {}, "ok": t.get("ok")} for t in tools
                    ],
                    "server_llm_ms": server.get("full_text_response_ms") or None,
                    "server_first_audio_ms": server.get("first_audio_heard_ms") or None,
                },
                "expected": {
                    "agent": agent or rule_route or None,
                    "action": tools[0]["tool"] if tools else "reply",
                    "llm_calls": 1 + len(tools) + (1 if rule_route is None else 0),
                    "notes": "",
                },
                "review": "pending",
            })

        # Append this turn to the running state the way the graph stores it.
        if user_text:
            history.append({"role": "user", "content": user_text})
        for n, call in enumerate(tools):
            history.extend(_tool_exchange(call, n, turn.get("turn_number") or i))
        if bot_text:
            history.append({"role": "assistant", "content": bot_text})
        if agent:
            prev_agent = agent
    return out


# ── slot variants ──────────────────────────────────────────────────────────

# The real logs have almost no lunch / off-grid / out-of-hours requests. These
# reuse a real state where the bot had just asked "what time?" and swap in an
# edge-case answer. Marked as variants: production never saw them.
SLOT_EDGE_PHRASES = [
    ("Can I come at 2:30 tomorrow?", "lunch clash 14:30"),
    ("How about 2 pm?", "lunch clash 14:00"),
    ("Tomorrow at 3:15 please.", "off-grid minute 15:15"),
    ("Can I do 10:45?", "off-grid minute 10:45"),
    ("8 in the morning works for me.", "outside hours 08:00"),
    ("Can I come at 6 pm after work?", "outside hours 18:00"),
    ("Around 5:30 in the evening.", "outside hours 17:30"),
    ("Is 1:55 free?", "off-grid minute 13:55"),
]
_ASKED_TIME_RE = re.compile(
    r"(what|which) (time|day)|preferred (time|day)|when would you like|time works|time would work"
    r"|morning or afternoon|any (particular )?(day|time)",
    re.IGNORECASE,
)


def slot_variants(snaps: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    bases = []
    seen_sessions: set[str] = set()
    for s in snaps:
        last_bot = next((m.get("content") or "" for m in reversed(s["history"]) if m["role"] == "assistant"
                         and m.get("content")), "")
        sid = s["source"]["session_id"]
        if _ASKED_TIME_RE.search(last_bot) and sid not in seen_sessions:
            bases.append(s)
            seen_sessions.add(sid)
    random.Random(seed).shuffle(bases)
    out = []
    for i, (phrase, reason) in enumerate(SLOT_EDGE_PHRASES[:count]):
        if not bases:
            break
        base = bases[i % len(bases)]
        rule_route = _sticky_route_from_history(_routing_view(base["history"], phrase), last_agent=base["prev_agent"])
        agent = base["expected"]["agent"] or rule_route or "booking"
        out.append({
            **json.loads(json.dumps(base)),
            "id": f"{base['id']}~slot{i + 1}",
            "categories": ["slot_edge"],
            "category": "slot_edge",
            "why": [reason, "variant of a real state (caller words replaced)"],
            "user_text": phrase,
            "rule_route": rule_route,
            "logged": {"agent": "", "reply": "", "tools": [], "server_llm_ms": None, "server_first_audio_ms": None},
            "expected": {
                "agent": agent,
                "action": "list_doctors",  # check real availability before answering
                "llm_calls": 2 + (1 if rule_route is None else 0),
                "notes": "variant: production never saw this turn; confirm the expected action",
            },
            "variant_of": base["id"],
        })
    return out


# ── selection ──────────────────────────────────────────────────────────────


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def select(snaps: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Fill each category quota with varied text from varied sessions."""
    rng = random.Random(seed)
    pool = snaps[:]
    rng.shuffle(pool)
    chosen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_text: set[str] = set()

    def take(cat: str, limit: int) -> None:
        per_session: Counter = Counter()
        for s in pool:
            if len([c for c in chosen if c["category"] == cat]) >= limit:
                return
            key = _norm(s["user_text"])
            sid = s["source"]["session_id"]
            if cat not in s["categories"] or s["id"] in used_ids or key in used_text:
                continue
            if per_session[sid] >= MAX_PER_SESSION_PER_CATEGORY:
                continue
            chosen.append({**s, "category": cat})
            used_ids.add(s["id"])
            used_text.add(key)
            per_session[sid] += 1

    for cat, share in QUOTAS.items():
        take(cat, max(1, round(size * share)))
    # Top up from any category if some quota ran short.
    for cat in QUOTAS:
        if len(chosen) >= size:
            break
        take(cat, len([c for c in chosen if c["category"] == cat]) + (size - len(chosen)))
    order = list(QUOTAS)
    chosen.sort(key=lambda c: (order.index(c["category"]), c["id"]))
    return chosen[:size]


def _keep_reviewed(items: list[dict[str, Any]], out_path: Path) -> int:
    """Carry over hand-checked labels from a previous build."""
    if not out_path.exists():
        return 0
    old = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("review") and item["review"] != "pending":
                old[item["id"]] = item
    kept = 0
    for item in items:
        prev = old.get(item["id"])
        if prev:
            item["expected"] = prev["expected"]
            item["review"] = prev["review"]
            kept += 1
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=50, help="number of snapshots (default 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slot-variants", type=int, default=8,
                    help=f"extra lunch / off-grid / out-of-hours variants (max {len(SLOT_EDGE_PHRASES)}, default 8)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    sessions = load_sessions()
    tool_calls = load_tool_calls()
    snaps = [s for sess in sessions for s in snapshots_for_session(sess, tool_calls.get(sess["session_id"], []))]
    items = select(snaps, args.size, args.seed)
    variants = slot_variants(snaps, args.slot_variants, args.seed)
    slot_end = max((i for i, it in enumerate(items) if it["category"] == "slot_edge"), default=-1) + 1
    items[slot_end:slot_end] = variants
    for n, item in enumerate(items, 1):
        item["n"] = n
    kept = _keep_reviewed(items, args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avail = Counter(c for s in snaps for c in s["categories"])
    picked = Counter(i["category"] for i in items)
    print(f"Real sessions: {len(sessions)}   turns: {len(snaps)}   "
          f"sessions with tool calls: {sum(1 for s in sessions if s['session_id'] in tool_calls)}")
    print(f"\n{'category':20s} {'available':>9s} {'picked':>7s}")
    for cat in QUOTAS:
        print(f"{cat:20s} {avail[cat]:9d} {picked[cat]:7d}")
    print(f"(slot_edge includes {len(variants)} variants of real states)")
    print(f"\nWrote {len(items)} snapshots -> {args.out.relative_to(ROOT)}"
          + (f"  (kept {kept} reviewed labels)" if kept else ""))
    print("\nSample:")
    for item in items[:: max(1, len(items) // 8)]:
        print(f"  [{item['category']}] prev={item['prev_agent'] or '-':13s} "
              f"{item['user_text'][:55]!r:60s} -> {item['expected']['agent']}/{item['expected']['action']}")
    print("\nNext: review 'expected' in the file (set review to \"ok\"), then run "
          "python eval/run_golden_latency.py")


if __name__ == "__main__":
    main()
