"""Go-live preflight: refuse a real-patient deployment until the HIPAA posture is complete.

    python go_live_check.py                 # production gate: is this safe for REAL patients?
    python go_live_check.py --profile dev   # dev gate: is the synthetic-data posture healthy?
    python go_live_check.py --json          # machine-readable

Run it in CI / before the first real patient record. It checks configuration only —
contracts (BAAs) are confirmed by the operator via the CONFIRMED_BAAS variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import security_config as sc

_ROOT = Path(__file__).resolve().parent

# Read the same .env the app loads, so the preflight sees the real configuration.
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
except Exception:  # noqa: BLE001
    pass


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def run_dev_checks() -> list[dict]:
    """Validate the synthetic-development posture: controls on, non-BAA vendors allowed."""
    out: list[dict] = []

    def add(name: str, ok: bool, detail: str, *, warn: bool = False) -> None:
        out.append({"check": name, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": detail})

    add("HIPAA_MODE on", sc.hipaa_mode(), "Keep HIPAA_MODE=1 even in dev so every control is exercised.")
    add("SYNTHETIC_DATA_ONLY on", sc.synthetic_data_only(),
        "Dev profile expects SYNTHETIC_DATA_ONLY=1 (no real PHI; non-BAA vendors permitted).")
    add("An LLM is usable", bool(sc.allowed_llm_providers()), "No provider available — check PHI_ALLOWED_PROVIDERS.")
    try:
        import admin_auth
        add("Admin account exists", bool(admin_auth.list_users()) or admin_auth.admin_configured(),
            "Create one: python manage_admin.py add <you>.")
    except Exception as exc:  # noqa: BLE001
        add("Admin account", False, str(exc))
    try:
        import audit_log
        add("Audit chain intact", bool(audit_log.verify_chain().get("ok")), "Audit log chain broken.")
    except Exception as exc:  # noqa: BLE001
        add("Audit chain", False, str(exc))
    add("Records file absent", not (_ROOT / "Records").exists(), "Delete Records / purge history before pushing.")
    add("DATA_DIR permissions", sc.data_dir() is None or not sc.data_dir().exists()
        or oct(sc.data_dir().stat().st_mode & 0o777) == "0o700",
        f"DATA_DIR should be 0700 when set (now {oct(sc.data_dir().stat().st_mode & 0o777) if (sc.data_dir() and sc.data_dir().exists()) else 'unset'}).", warn=True)
    out.append({"check": "Reminder", "status": "WARN",
                "detail": "This is the DEV gate. Run `python go_live_check.py` (no --profile) before real patient data."})
    return out


def run_checks() -> list[dict]:
    out: list[dict] = []
    synthetic = sc.synthetic_data_only()

    def add(name: str, ok: bool, detail: str, *, warn: bool = False, prod_only: bool = False) -> None:
        # prod_only items are advisory while developing on synthetic data, hard fails before real PHI.
        status = "PASS" if ok else ("WARN" if (warn or (prod_only and synthetic)) else "FAIL")
        out.append({"check": name, "status": status, "detail": detail})

    add("HIPAA_MODE on", sc.hipaa_mode(), "HIPAA_MODE must be 1 (default).")
    add(
        "SYNTHETIC_DATA_ONLY off",
        not sc.synthetic_data_only(),
        "SYNTHETIC_DATA_ONLY must be unset/0 before real data (it admits non-BAA vendors/channels).",
    )
    allowed = sc.allowed_llm_providers()
    add(
        "LLM allow-list is BAA-only",
        bool(allowed) and allowed <= sc.BAA_CAPABLE_LLM_PROVIDERS,
        f"PHI_ALLOWED_PROVIDERS resolves to {sorted(allowed) or 'none'}; must be a non-empty subset of "
        f"{sorted(sc.BAA_CAPABLE_LLM_PROVIDERS)}.",
    )
    confirmed = {p.strip().lower() for p in _env("CONFIRMED_BAAS").split(",") if p.strip()}
    required = set(allowed)
    if _env("DEEPGRAM_API_KEY"):
        required.add("deepgram")
    add(
        "BAAs confirmed for every PHI vendor",
        bool(allowed) and required <= confirmed,
        f"Needed: {sorted(required) or 'none'}; CONFIRMED_BAAS now: {sorted(confirmed) or 'none'}. "
        "Set CONFIRMED_BAAS=openai[,groq],deepgram[,daily,hosting] once the signed agreements are in hand.",
        prod_only=True,
    )
    for key in ("DEEPSEEK_API_KEY", "GLM_API_KEY", "GOOGLE_API_KEY"):
        add(f"{key} absent", not _env(key), f"Remove {key} — that vendor cannot receive PHI.")
    add("Telegram is non-PHI", not sc.channel_may_carry_phi("telegram"), "Telegram must be a hand-off channel (NON_PHI_CHANNELS).")
    add("Discord is non-PHI", not sc.channel_may_carry_phi("discord"), "Discord must be a hand-off channel (NON_PHI_CHANNELS).")
    add("DATA_DIR set", sc.data_dir() is not None, "DATA_DIR must point at an encrypted volume.", prod_only=True)
    if sc.data_dir():
        mode = oct(sc.data_dir().stat().st_mode & 0o777) if sc.data_dir().exists() else "missing"
        add("DATA_DIR owner-only", mode == "0o700", f"DATA_DIR mode is {mode}; expected 0o700.")
    add("Secure cookies", sc.cookie_secure(), "COOKIE_SECURE must be on (HTTPS in front of the app).")
    add(
        "ALLOWED_ORIGINS explicit",
        bool(_env("ALLOWED_ORIGINS")) and "localhost" not in _env("ALLOWED_ORIGINS"),
        "Set ALLOWED_ORIGINS to the production origin(s) (localhost is fine only for synthetic-data dev).",
        prod_only=True,
    )
    add("ADMIN_TOKEN_SECRET set", bool(_env("ADMIN_TOKEN_SECRET")) and _env("ADMIN_TOKEN_SECRET") != "change-me-in-production",
        "Set ADMIN_TOKEN_SECRET to a long random value (or rely on the 0600 secret file).", warn=True)
    try:
        import admin_auth

        users = admin_auth.list_users()
        add("Admin accounts exist (hashed)", bool(users), f"{len(users)} account(s) in admin_users.json; create with manage_admin.py.")
        add("No env ADMIN_PASS", not _env("ADMIN_PASS"), "Remove ADMIN_PASS from the environment; use hashed accounts.")
    except Exception as exc:  # noqa: BLE001
        add("Admin accounts", False, f"could not read admin accounts: {exc}")
    add("BACKUP_ENC_KEY set", bool(_env("BACKUP_ENC_KEY")), "Backups must be encrypted (python backup.py keygen).")
    transport = _env("MCP_TRANSPORT") or "stdio"
    add("MCP auth", transport == "stdio" or bool(sc.mcp_api_key()), f"MCP_TRANSPORT={transport}: HTTP/SSE needs MCP_API_KEY.")
    add("LLM QA dumps", not sc._env_bool("LLM_QA_DUMPS", False),
        "LLM_QA_DUMPS=1 keeps raw prompts for QA (admin-only, purged after RETAIN_LLM_DUMPS_DAYS). Acceptable if documented.", warn=True)
    add("MCP tunnel guard on", _env("MCP_ALLOW_TUNNEL").lower() not in ("1", "true", "yes"), "MCP_ALLOW_TUNNEL must be off; use MCP_ALLOWED_HOSTS.")
    add("Records file absent", not (_ROOT / "Records").exists(), "Delete Records and purge it from git history (scripts/purge_records_history.sh).")
    try:
        import audit_log

        chain = audit_log.verify_chain()
        add("Audit chain intact", bool(chain.get("ok")), f"{chain}")
    except Exception as exc:  # noqa: BLE001
        add("Audit chain", False, str(exc))
    try:
        from service_settings import load_settings

        s = load_settings()
        qa = sc._env_bool("LLM_QA_DUMPS", False)
        add("Raw LLM dumps off", (not s.get("save_llm_messages")) or qa,
            "save_llm_messages must be false unless LLM_QA_DUMPS=1 (documented QA exception).", warn=qa)
        add("Cascade LLM allowed", s.get("cascade_llm") in allowed, f"cascade_llm={s.get('cascade_llm')}")
    except Exception as exc:  # noqa: BLE001
        add("Admin settings", False, str(exc), warn=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HIPAA go-live preflight")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile", choices=("prod", "dev"), default="prod",
                        help="prod (default): real-patient gate; dev: synthetic-development gate.")
    args = parser.parse_args(argv)
    results = run_dev_checks() if args.profile == "dev" else run_checks()
    fails = [r for r in results if r["status"] == "FAIL"]
    if args.json:
        print(json.dumps({"ok": not fails, "results": results}, indent=2))
    else:
        width = max(len(r["check"]) for r in results)
        for r in results:
            print(f"[{r['status']:4}] {r['check']:<{width}}  {r['detail']}")
        print()
        label = "DEV POSTURE" if args.profile == "dev" else "GO-LIVE"
        print(f"{label}: " + ("OK — configuration is healthy." if not fails else f"BLOCKED — {len(fails)} failing check(s)."))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
