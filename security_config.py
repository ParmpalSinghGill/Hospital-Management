"""Central security / HIPAA-mode configuration shared by every entry point.

Everything PHI-related is decided here so the rules live in one place:

* ``HIPAA_MODE`` (default **on**) — fail-closed posture: PHI may only go to
  allow-listed vendors, debug dumps are disabled, Telegram/Discord become
  non-PHI hand-off channels, secure cookies are enforced, etc.
  Set ``HIPAA_MODE=0`` only for local development with synthetic data.
* ``DATA_DIR`` — optional single root for every PHI store (db, chats,
  metrics, logs, audit, backups) so it can be placed on an encrypted volume.
  When unset the historical in-repo locations are used.
* Secrets, allowed origins, provider allow-list, cookie flags.

Import this module early (before loguru/logging sinks are created).
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def hipaa_mode() -> bool:
    """True unless ``HIPAA_MODE`` is explicitly turned off."""
    return _env_bool("HIPAA_MODE", True)


def synthetic_data_only() -> bool:
    """Development posture: every HIPAA control stays ON, but because the database holds
    only synthetic (MakeDataBase.py) records, vendors/channels without a BAA may be used.

    Set ``SYNTHETIC_DATA_ONLY=1`` while building with DeepSeek/GLM/Telegram/Discord.
    It MUST be unset (or 0) before a single real patient record enters the system —
    ``python go_live_check.py`` enforces that.
    """
    return hipaa_mode() and _env_bool("SYNTHETIC_DATA_ONLY", False)


# ---------------------------------------------------------------- paths ----
def data_dir() -> Path | None:
    raw = (os.getenv("DATA_DIR") or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _under(sub: str, legacy: Path) -> Path:
    base = data_dir()
    return (base / sub) if base else legacy


def dataset_dir() -> Path:
    return _under("dataset", _ROOT / "dataset")


def chats_dir() -> Path:
    env = (os.getenv("CHAT_LOGS_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return _under("chats", _ROOT / "chats")


def metrics_dir() -> Path:
    return _under("metrics", _ROOT / "metrics")


def logs_dir() -> Path:
    base = data_dir()
    return (base / "logs") if base else _ROOT


def app_log_path() -> Path:
    d = logs_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        if data_dir():
            d.chmod(0o700)
    except OSError:
        pass
    return d / "app.log"


def audit_dir() -> Path:
    return _under("audit", _ROOT / "audit")


def backups_dir() -> Path:
    return _under("backups", _ROOT / "backups")


def admin_users_path() -> Path:
    return _under("admin_users.json", _ROOT / "admin_users.json")


def secret_file_path() -> Path:
    return _under(".admin_token_secret", _ROOT / ".admin_token_secret")


# -------------------------------------------------------------- secrets ----
_SECRET_CACHE: str | None = None


def server_secret() -> str:
    """Stable per-deployment secret (env ``ADMIN_TOKEN_SECRET`` or a 0600 file)."""
    global _SECRET_CACHE
    if _SECRET_CACHE:
        return _SECRET_CACHE
    env = (os.getenv("ADMIN_TOKEN_SECRET") or "").strip()
    if env and env != "change-me-in-production":
        _SECRET_CACHE = env
        return env
    path = secret_file_path()
    try:
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                _SECRET_CACHE = val
                _chmod_quiet(path, 0o600)
                return val
    except OSError:
        pass
    val = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_private(path, val)
    except OSError:
        pass
    _SECRET_CACHE = val
    return val


# ------------------------------------------------------------ providers ----
# Vendors that will sign a BAA (see MD/hipaa/02-security-risk-assessment.md §7).
# Only these may receive PHI. DeepSeek / GLM are never allowed in HIPAA mode.
BAA_CAPABLE_LLM_PROVIDERS: frozenset[str] = frozenset({"openai", "groq"})
NEVER_PHI_PROVIDERS: frozenset[str] = frozenset({"deepseek", "glm"})


def allowed_llm_providers() -> frozenset[str]:
    """Providers that may receive PHI right now.

    In HIPAA mode: ``PHI_ALLOWED_PROVIDERS`` (comma list, default ``openai``)
    intersected with the BAA-capable set. Outside HIPAA mode: everything.
    """
    all_providers = frozenset({"deepseek", "glm", "groq", "openai"})
    if not hipaa_mode():
        return all_providers
    if synthetic_data_only():
        # No PHI exists yet: any provider listed (default: all) — controls stay on.
        raw = (os.getenv("PHI_ALLOWED_PROVIDERS") or ",".join(sorted(all_providers))).lower()
        return frozenset(p.strip() for p in raw.split(",") if p.strip()) & all_providers
    raw = (os.getenv("PHI_ALLOWED_PROVIDERS") or "openai").lower()
    wanted = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(wanted & BAA_CAPABLE_LLM_PROVIDERS)


def provider_allowed(provider: str) -> bool:
    return (provider or "").strip().lower() in allowed_llm_providers()


def provider_block_reason(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in NEVER_PHI_PROVIDERS:
        return f"{p} offers no HIPAA BAA and cannot receive PHI (HIPAA_MODE is on)."
    return (
        f"{p} is not in PHI_ALLOWED_PROVIDERS. Add it only after a BAA is executed "
        "(HIPAA_MODE is on)."
    )


# ------------------------------------------------------------- channels ----
def non_phi_channels() -> frozenset[str]:
    """Channels that may NOT carry PHI (no BAA possible): hand-off only."""
    if not hipaa_mode():
        return frozenset()
    if synthetic_data_only() and os.getenv("NON_PHI_CHANNELS") is None:
        return frozenset()  # synthetic data: bots may be exercised end-to-end
    raw = os.getenv("NON_PHI_CHANNELS", "telegram,discord")
    return frozenset(c.strip().lower() for c in raw.split(",") if c.strip())


def channel_may_carry_phi(channel: str) -> bool:
    return (channel or "").strip().lower() not in non_phi_channels()


def web_app_url() -> str:
    return (os.getenv("WEB_APP_URL") or "").strip() or "the clinic's secure web assistant"


def handoff_message() -> str:
    return (
        "For your privacy, I can't discuss appointments or personal details on this "
        f"channel. Please use {web_app_url()} or call the front desk, and we'll take "
        "care of it right away."
    )


# ------------------------------------------------------------- web/http ----
def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    vals = [o.strip() for o in raw.split(",") if o.strip()]
    if vals:
        return vals
    return ["http://localhost:7860", "http://127.0.0.1:7860"]


def cookie_secure() -> bool:
    """Send admin cookies only over HTTPS (default: on in HIPAA mode)."""
    return _env_bool("COOKIE_SECURE", hipaa_mode())


def debug_dumps_allowed() -> bool:
    """May raw LLM prompt/response dumps (chats/messages/) be written?

    * HIPAA_MODE off → yes.
    * SYNTHETIC_DATA_ONLY → yes (no real PHI; needed for prompt/QA work).
    * Production → only with the explicit ``LLM_QA_DUMPS=1`` (admin-only, audited,
      purged by retention after RETAIN_LLM_DUMPS_DAYS, default 7).
    """
    if not hipaa_mode() or synthetic_data_only():
        return True
    return _env_bool("LLM_QA_DUMPS", False)


def mcp_api_key() -> str:
    return (os.getenv("MCP_API_KEY") or "").strip()


# ------------------------------------------------------------ hardening ----
def _chmod_quiet(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass


def _write_private(path: Path, text: str) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def harden_permissions(extra_files: Iterable[Path] = (), extra_dirs: Iterable[Path] = ()) -> None:
    """Owner-only permissions on every PHI store and secret; set process umask.

    Safe to call at every start-up; failures are ignored (read-only mounts).
    """
    try:
        os.umask(0o077)
    except Exception:  # noqa: BLE001
        pass
    dirs = [dataset_dir(), chats_dir(), metrics_dir(), audit_dir(), backups_dir(), *extra_dirs]
    if data_dir():
        dirs.append(logs_dir())
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        _chmod_quiet(d, 0o700)
    files = [
        _ROOT / ".env",
        secret_file_path(),
        admin_users_path(),
        dataset_dir() / "hospital.db",
        app_log_path(),
        *extra_files,
    ]
    for f in files:
        if f.exists() and stat.S_ISREG(f.stat().st_mode):
            _chmod_quiet(f, 0o600)


def startup_warnings() -> list[str]:
    """Human-readable posture warnings for operators (logged at start)."""
    out: list[str] = []
    if not hipaa_mode():
        out.append("HIPAA_MODE is OFF — only use with synthetic data.")
        return out
    if synthetic_data_only():
        out.append(
            "SYNTHETIC_DATA_ONLY=1 — non-BAA vendors/channels are permitted because the data is "
            "synthetic. Unset it (and run go_live_check.py) before any real patient data."
        )
    if not data_dir():
        out.append(
            "DATA_DIR is unset — PHI stores live inside the repo directory. "
            "Point DATA_DIR at an encrypted volume for production."
        )
    if not allowed_llm_providers():
        out.append(
            "PHI_ALLOWED_PROVIDERS resolves to an empty set — no LLM can be used. "
            "Set it to a BAA-covered provider (e.g. openai)."
        )
    if not cookie_secure():
        out.append("COOKIE_SECURE is off — admin cookies may travel over plain HTTP.")
    if not synthetic_data_only() and _env_bool("LLM_QA_DUMPS", False):
        out.append("LLM_QA_DUMPS=1 — raw prompt/response dumps are being kept for QA (admin-only, short retention).")
    return out
