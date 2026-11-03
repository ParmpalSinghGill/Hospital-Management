"""Replay the golden dataset: one LLM call per snapshot, timed like production.

For every snapshot in ``eval/golden/golden_turns.jsonl`` this sends exactly what
the specialist agent would send at that moment — its system prompt, the
conversation so far (including earlier tool calls/results), the caller's new
words, and the agent's tools — as ONE chat call. Nothing is executed: the
model's first decision (call a tool, or reply) is recorded and compared with
the expected action.

Timing comes from the production transport (``llm_transport.py`` +
``llm_timing.py``), so the numbers mean the same as in live calls:

  wait         request sent -> first response byte   (network + provider queue)
  first token  first byte   -> first token           (model reading the prompt)
  stream       first token  -> last token            (writing the answer)
  total        request sent -> last token

At the end it adds the measured speech parts (end-of-speech wait, STT, TTS from
``metrics/bot_messages.csv``) to estimate caller-stops -> bot-audio latency.

    python eval/run_golden_latency.py                       # provider from admin settings, 3 repeats
    python eval/run_golden_latency.py --provider openai --repeats 5
    python eval/run_golden_latency.py --agent booking       # every snapshot on one agent (no-router test)
    LLM_HEDGE=1 python eval/run_golden_latency.py --label hedge3

Outputs (metrics/golden/):  <label>_calls.csv  (one row per call)
                            <label>_summary.json (medians, accuracy, speech-to-speech estimate)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

GOLDEN = ROOT / "eval" / "golden" / "golden_turns.jsonl"
OUT_DIR = ROOT / "metrics" / "golden"
BOT_MESSAGES = ROOT / "metrics" / "bot_messages.csv"
ADMIN_SETTINGS = ROOT / "admin_settings.json"

AGENT_BUILDERS = {
    "general": "build_general_agent",
    "booking": "build_booking_agent",
    "cancelling": "build_cancellation_agent",
    "rescheduling": "build_reschedule_agent",
    "prescriptions": "build_prescription_agent",
    "router": "build_router_agent",
}


# ── helpers ────────────────────────────────────────────────────────────────


def pct(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def med(values: list[float]) -> float | None:
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def fmt(v: float | None, unit: str = "") -> str:
    return "-" if v is None else f"{v:,.0f}{unit}"


def _settings() -> dict[str, Any]:
    try:
        return json.loads(ADMIN_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_messages(item: dict[str, Any], system_prompt: str) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    msgs: list[Any] = [SystemMessage(system_prompt)]
    for m in item["history"]:
        role = m["role"]
        if role == "user":
            msgs.append(HumanMessage(m["content"]))
        elif role == "assistant" and m.get("tool_calls"):
            msgs.append(AIMessage(content=m.get("content") or "", tool_calls=[
                {"id": tc["id"], "name": tc["name"], "args": tc.get("args") or {}, "type": "tool_call"}
                for tc in m["tool_calls"]
            ]))
        elif role == "assistant":
            msgs.append(AIMessage(m.get("content") or ""))
        elif role == "tool":
            msgs.append(ToolMessage(content=m.get("content") or "", tool_call_id=m["tool_call_id"]))
    msgs.append(HumanMessage(item["user_text"]))
    return msgs


def build_agents(model: Any) -> dict[str, tuple[str, Any]]:
    """name -> (system prompt, model with that agent's tools bound), from production code."""
    import reception_agents as ra

    out = {}
    for name, builder in AGENT_BUILDERS.items():
        agent = getattr(ra, builder)()
        tool_node = agent.nodes.get("tools")
        tools = list(tool_node.bound.tools_by_name.values()) if tool_node else []
        bound = model.bind_tools(tools) if tools else model
        out[name] = (ra.AGENT_SYSTEM_PROMPTS[name], bound)
    return out


def speech_parts(vad_override: float | None, stt_override: float | None, tts_override: float | None) -> dict:
    """Median non-LLM parts of a voice turn from live cascade calls."""
    stt, tts = [], []
    if BOT_MESSAGES.exists():
        with BOT_MESSAGES.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("pipeline_mode") != "cascade":
                    continue
                try:
                    if r.get("stt_latency_ms"):
                        stt.append(float(r["stt_latency_ms"]))
                    if r.get("first_audio_heard_ms") and r.get("full_text_response_ms"):
                        gap = float(r["first_audio_heard_ms"]) - float(r["full_text_response_ms"])
                        if gap >= 0:
                            tts.append(gap)
                except ValueError:
                    continue
    vad = float(_settings().get("vad_stop_secs") or 0.2) * 1000
    return {
        "end_of_speech_wait_ms": vad_override if vad_override is not None else vad,
        "stt_final_ms": stt_override if stt_override is not None else (med(stt) or 50.0),
        "tts_first_audio_ms": tts_override if tts_override is not None else (med(tts) or 380.0),
        "samples": {"stt": len(stt), "tts": len(tts)},
    }


# ── run ────────────────────────────────────────────────────────────────────


def call_once(llm_timing: Any, bound: Any, messages: list[Any], scope_id: str) -> dict[str, Any]:
    llm_timing.set_scope(scope_id)
    t0 = time.perf_counter()
    resp = bound.invoke(messages)
    wall = (time.perf_counter() - t0) * 1000
    calls = [c for c in llm_timing.drain(scope_id) if c.total_ms is not None]
    win = min(calls, key=lambda c: c.t4 or 0.0) if calls else None
    usage = getattr(resp, "usage_metadata", None) or {}
    token_usage = (getattr(resp, "response_metadata", None) or {}).get("token_usage") or {}
    tool_calls = getattr(resp, "tool_calls", None) or []
    return {
        "wall_ms": round(wall, 1),
        "wait_ms": win.wait_ms if win else None,
        "first_token_ms": win.first_token_ms if win else None,
        "stream_ms": win.stream_ms if win else None,
        "total_ms": win.total_ms if win else round(wall, 1),
        "copies": len(calls),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cached_tokens": token_usage.get("prompt_cache_hit_tokens")
        or (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "action": tool_calls[0]["name"] if tool_calls else "reply",
        "reply": (resp.content or "")[:160] if isinstance(resp.content, str) else "",
    }


def summarize(rows: list[dict[str, Any]], items: list[dict[str, Any]], speech: dict, args: Any) -> dict:
    ok = [r for r in rows if not r.get("error")]

    def block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "calls": len(rs),
            "total_median": med([r["total_ms"] for r in rs]),
            "total_p90": pct([r["total_ms"] for r in rs], 0.9),
            "wait_median": med([r["wait_ms"] for r in rs]),
            "first_token_median": med([r["first_token_ms"] for r in rs]),
            "stream_median": med([r["stream_ms"] for r in rs]),
            "input_tokens_median": med([r["input_tokens"] for r in rs]),
            "action_match": (sum(r["match"] for r in rs) / len(rs)) if rs else None,
        }

    by_cat = {c: block([r for r in ok if r["category"] == c]) for c in dict.fromkeys(r["category"] for r in ok)}
    by_agent = {a: block([r for r in ok if r["agent"] == a]) for a in dict.fromkeys(r["agent"] for r in ok)}
    reviewed = [r for r in ok if r["review"] != "pending"]

    per_call = med([r["total_ms"] for r in ok]) or 0.0
    calls_per_turn = statistics.mean(i["expected"].get("llm_calls", 1) for i in items) if items else 1.0
    fixed = speech["end_of_speech_wait_ms"] + speech["stt_final_ms"] + speech["tts_first_audio_ms"]
    s2s = {
        **speech,
        "llm_call_median_ms": per_call,
        "llm_calls_per_turn_mean": round(calls_per_turn, 2),
        "one_call_turn_ms": fixed + per_call,
        "two_call_turn_ms": fixed + 2 * per_call,
        "mixed_turn_ms": fixed + calls_per_turn * per_call,
        "after_turn_end_one_call_ms": fixed - speech["end_of_speech_wait_ms"] + per_call,
    }
    return {
        "label": args.label,
        "provider": args.provider,
        "agent_mode": args.agent,
        "repeats": args.repeats,
        "hedge": os.getenv("LLM_HEDGE", ""),
        "http2": os.getenv("LLM_HTTP2", ""),
        "stream": os.getenv("LLM_STREAM", "1"),
        "snapshots": len(items),
        "errors": len(rows) - len(ok),
        "overall": block(ok),
        "reviewed_only": block(reviewed) if reviewed else None,
        "by_category": by_cat,
        "by_agent": by_agent,
        "speech_to_speech": s2s,
    }


def print_report(s: dict[str, Any]) -> None:
    o = s["overall"]
    print(f"\n=== {s['label']}  provider={s['provider']}  agent={s['agent_mode']}  "
          f"{s['snapshots']} snapshots x {s['repeats']}  errors={s['errors']} ===\n")
    head = f"| {'group':22s} | calls | total p50 | total p90 | wait | prompt read | stream | in tok | action ok |"
    print(head)
    print("|" + "|".join("-" * (len(c)) for c in head.split("|")[1:-1]) + "|")

    def line(name: str, b: dict[str, Any]) -> None:
        acc = "-" if b["action_match"] is None else f"{b['action_match'] * 100:.0f}%"
        print(f"| {name:22s} | {b['calls']:5d} | {fmt(b['total_median']):>9s} | {fmt(b['total_p90']):>9s} | "
              f"{fmt(b['wait_median']):>4s} | {fmt(b['first_token_median']):>11s} | {fmt(b['stream_median']):>6s} | "
              f"{fmt(b['input_tokens_median']):>6s} | {acc:>9s} |")

    line("ALL", o)
    for k, b in s["by_category"].items():
        line(f"cat: {k}", b)
    for k, b in s["by_agent"].items():
        line(f"agent: {k}", b)
    if s["reviewed_only"]:
        line("reviewed labels only", s["reviewed_only"])

    x = s["speech_to_speech"]
    print("\nSpeech-to-speech estimate (caller stops talking -> bot audio starts):")
    print(f"  end-of-speech wait (setting)   {fmt(x['end_of_speech_wait_ms'], ' ms')}")
    print(f"  STT final transcript (median)  {fmt(x['stt_final_ms'], ' ms')}   ({x['samples']['stt']} live turns)")
    print(f"  LLM per call (median, golden)  {fmt(x['llm_call_median_ms'], ' ms')}")
    print(f"  TTS first audio (median)       {fmt(x['tts_first_audio_ms'], ' ms')}   ({x['samples']['tts']} live turns)")
    print(f"  -> turn with 1 LLM call        {fmt(x['one_call_turn_ms'], ' ms')}")
    print(f"  -> turn with 2 LLM calls       {fmt(x['two_call_turn_ms'], ' ms')}   (tool round or router LLM)")
    print(f"  -> golden mix ({x['llm_calls_per_turn_mean']} calls/turn)   {fmt(x['mixed_turn_ms'], ' ms')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--provider", default=None, help="deepseek | openai | groq | glm (default: admin cascade_llm)")
    ap.add_argument("--agent", default="logged",
                    help="'logged' = the agent in the label (default), or force one: " + ", ".join(AGENT_BUILDERS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N snapshots")
    ap.add_argument("--category", default="", help="only this category")
    ap.add_argument("--warmup", type=int, default=1, help="untimed calls first (opens the connection)")
    ap.add_argument("--label", default="")
    ap.add_argument("--vad-ms", type=float, default=None, help="override end-of-speech wait")
    ap.add_argument("--stt-ms", type=float, default=None, help="override STT median")
    ap.add_argument("--tts-ms", type=float, default=None, help="override TTS first-audio median")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    args.provider = (args.provider or os.getenv("LLM_PROVIDER") or _settings().get("cascade_llm") or "deepseek").lower()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.label = args.label or f"{args.provider}-{args.agent}-{stamp}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Keep golden calls out of the live metrics/llm_calls.csv (read at import time).
    os.environ["LLM_CALLS_CSV"] = str(OUT_DIR / f"{args.label}_llm_calls_raw.csv")

    import llm_timing
    from Model import _init_model

    items = load_golden(args.golden)
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no snapshots selected")

    model = _init_model(callbacks=None, provider=args.provider)
    agents = build_agents(model)
    if args.agent != "logged" and args.agent not in agents:
        raise SystemExit(f"unknown agent {args.agent!r}")

    def agent_for(item: dict[str, Any]) -> str:
        if args.agent != "logged":
            return args.agent
        name = item["expected"].get("agent") or item.get("rule_route") or "general"
        return name if name in agents else "general"

    for i in range(args.warmup):
        prompt, bound = agents[agent_for(items[0])]
        try:
            call_once(llm_timing, bound, to_messages(items[0], prompt), f"golden-warmup-{i}")
        except Exception as exc:  # noqa: BLE001
            print(f"warm-up failed: {exc}")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    total = len(items) * args.repeats
    done = 0
    for rep in range(args.repeats):
        order = items[:]
        rng.shuffle(order)  # spread provider drift across snapshots
        for item in order:
            name = agent_for(item)
            prompt, bound = agents[name]
            expected = item["expected"].get("action") or "reply"
            row = {
                "id": item["id"], "n": item.get("n"), "category": item["category"], "agent": name,
                "rep": rep, "review": item.get("review", "pending"), "expected_action": expected,
                "user_text": item["user_text"][:80], "history_messages": len(item["history"]),
            }
            try:
                row.update(call_once(llm_timing, bound, to_messages(item, prompt), f"golden-{item['id']}-{rep}"))
                row["match"] = int(row["action"] == expected)
            except Exception as exc:  # noqa: BLE001
                row.update({"error": f"{type(exc).__name__}: {exc}"[:300], "match": 0})
            rows.append(row)
            done += 1
            status = row.get("error") or f"{fmt(row.get('total_ms'))} ms  {row.get('action')}"
            print(f"[{done:3d}/{total}] #{item.get('n', '?'):>2} {item['category'][:12]:12s} {name[:12]:12s} {status}")

    speech = speech_parts(args.vad_ms, args.stt_ms, args.tts_ms)
    summary = summarize(rows, items, speech, args)
    print_report(summary)

    calls_csv = OUT_DIR / f"{args.label}_calls.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with calls_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {calls_csv.relative_to(ROOT)} and {args.label}_summary.json")


if __name__ == "__main__":
    main()
