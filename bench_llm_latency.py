"""Benchmark LLM call latency: HTTP/1.1 vs HTTP/2, single vs hedged requests.

Uses the real booking-agent system prompt + tools, so the payload matches a
Cascade voice turn. Configs are interleaved in random order every round so
provider-side load drifts hit all of them equally.

    python bench_llm_latency.py --provider deepseek --rounds 15
    python bench_llm_latency.py --provider openai --rounds 15

Results: printed table + metrics/llm_bench.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from Model import _deepseek_base_url, _deepseek_model_name, _hedged_classes, _openai_chat_model  # noqa: E402

# (name, http2, hedge_count, hedge_delay_ms); None http2 = untouched LangChain default (old code path)
CONFIGS = [
    ("baseline (old code)", None, 1, 0),
    ("http1.1 x1", False, 1, 0),
    ("http2 x1", True, 1, 0),
    ("http1.1 hedge x2", False, 2, 0),
    ("http2 hedge x2", True, 2, 0),
    ("http2 hedge x3", True, 3, 0),
    ("http2 hedge x2 +400ms", True, 2, 400),
]

USER_TURNS = [
    "I'd like to book an appointment with a cardiologist tomorrow.",
    "Hi, I have a bad toothache, can I see someone this week?",
    "Which doctors are available on Friday?",
    "My stomach hurts, I need to see a doctor.",
]


def provider_kwargs(provider: str) -> dict:
    if provider == "deepseek":
        return dict(model=_deepseek_model_name(), api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=_deepseek_base_url(), model_kwargs={"parallel_tool_calls": False})
    if provider == "openai":
        return dict(model=_openai_chat_model(), api_key=os.environ["OPENAI_API_KEY"])
    raise SystemExit(f"unsupported provider {provider!r} (deepseek | openai)")


def booking_payload():
    from reception_agents import AGENT_SYSTEM_PROMPTS, build_booking_agent
    from Tools import book_appointment, list_doctors, lookup_patient, save_patient

    build_booking_agent()
    return AGENT_SYSTEM_PROMPTS["booking"], [list_doctors, lookup_patient, save_patient, book_appointment]


def build_models(provider: str, tools):
    hedged_cls, _ = _hedged_classes()
    base = dict(temperature=0, **provider_kwargs(provider))
    models, clients = {}, {}
    for name, h2, n, delay in CONFIGS:
        if h2 is None:
            m = ChatOpenAI(**base)
        else:
            # Fresh client per config so connection reuse is measured fairly.
            clients[name] = httpx.AsyncClient(http2=h2, timeout=httpx.Timeout(60.0, connect=10.0))
            m = hedged_cls(**base, http_async_client=clients[name], hedge_count=n, hedge_delay_ms=delay)
        models[name] = m.bind_tools(tools)
    return models, clients


def pct(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    system, tools = booking_payload()
    models, clients = build_models(args.provider, tools)
    cold: dict[str, float] = {}
    warm: dict[str, list[float]] = {name: [] for name in models}
    errors: dict[str, int] = {name: 0 for name in models}
    rows = []

    for r in range(args.rounds + 1):  # round 0 = cold (first call opens the connection)
        order = list(models)
        random.shuffle(order)
        user = USER_TURNS[r % len(USER_TURNS)]
        for name in order:
            msgs = [SystemMessage(system), HumanMessage(user)]
            t = time.perf_counter()
            try:
                models[name].invoke(msgs)
                ms = (time.perf_counter() - t) * 1000
            except Exception as exc:  # noqa: BLE001
                errors[name] += 1
                print(f"  ! {name}: {type(exc).__name__}: {exc}")
                continue
            (cold.__setitem__(name, ms) if r == 0 else warm[name].append(ms))
            rows.append({"provider": args.provider, "config": name, "round": r, "ms": round(ms)})
        print(f"round {r}/{args.rounds} done")

    base_p50 = statistics.median(warm[CONFIGS[0][0]])
    print(f"\nProvider: {args.provider}  model: {provider_kwargs(args.provider)['model']}  "
          f"warm rounds: {args.rounds}\n")
    print(f"| {'config':24s} | cold ms | p50 ms | p90 ms |  max ms | mean ms | vs baseline p50 | conns | err |")
    print(f"|{'-'*26}|--------:|-------:|-------:|--------:|--------:|----------------:|------:|----:|")
    for name, *_ in CONFIGS:
        v = warm[name]
        if not v:
            continue
        conns = "-"
        if name in clients:
            conns = str(len(clients[name]._transport._pool.connections))
        p50 = statistics.median(v)
        print(f"| {name:24s} | {cold.get(name, float('nan')):7.0f} | {p50:6.0f} | {pct(v, .9):6.0f} | "
              f"{max(v):7.0f} | {statistics.mean(v):7.0f} | {p50 - base_p50:+15.0f} | {conns:>5s} | {errors[name]:3d} |")

    out = Path(__file__).parent / "metrics" / "llm_bench.csv"
    new = not out.exists()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["at", "provider", "config", "round", "ms"])
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({"at": stamp, **row})
    print(f"\nRaw timings appended to {out}")


if __name__ == "__main__":
    main()
