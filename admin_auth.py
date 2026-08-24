"""Admin authentication: per-user accounts, scrypt password hashes, roles,
server-side sessions with expiry/revocation, login lockout.

User store: ``admin_users.json`` (0600, gitignored) —
``{"users": [{"username": "...", "hash": "scrypt$<salt>$<hex>", "role": "admin|viewer"}]}``.
Manage with ``python manage_admin.py add <user> [--role viewer]``.

Fallback (single account) from env ``ADMIN_USER`` + ``ADMIN_PASS`` or
``ADMIN_PASS_HASH``; weak/default passwords are **refused** in HIPAA mode.

Session token = ``<session_id>.<hmac>``; the session id is looked up server-side
so logout / restart invalidates it, and the HMAC stops forged ids.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import security_config as sc
from rate_limit import Lockout, RateLimiter

ROLES = ("admin", "viewer")
WEAK_PASSWORDS = {"", "12345", "123456", "password", "admin", "changeme", "change-me"}

IDLE_TIMEOUT_SECS = int(os.getenv("ADMIN_IDLE_TIMEOUT_SECS", str(30 * 60)))
ABSOLUTE_TIMEOUT_SECS = int(os.getenv("ADMIN_ABSOLUTE_TIMEOUT_SECS", str(12 * 3600)))

_LOCKOUT = Lockout(max_failures=5, window_secs=15 * 60, lock_secs=15 * 60)
_LOGIN_RATE = RateLimiter(max_events=20, window_secs=60)


# ------------------------------------------------------------ passwords ----
def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest_hex = (stored or "").split("$", 2)
        if algo != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def is_weak_password(password: str) -> bool:
    p = password or ""
    return p.strip().lower() in WEAK_PASSWORDS or len(p) < 10


# ----------------------------------------------------------- user store ----
@dataclass
class AdminUser:
    username: str
    role: str = "admin"
    hash: str = ""


def _users_path() -> Path:
    return sc.admin_users_path()


def _load_store() -> dict[str, Any]:
    p = _users_path()
    if not p.exists():
        return {"users": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("users"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"users": []}


def _save_store(data: dict[str, Any]) -> None:
    p = _users_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, p)
    try:
        p.chmod(0o600)
    except OSError:
        pass


def list_users() -> list[AdminUser]:
    out = []
    for u in _load_store().get("users", []):
        if isinstance(u, dict) and u.get("username"):
            out.append(AdminUser(str(u["username"]), str(u.get("role") or "admin"), str(u.get("hash") or "")))
    return out


def upsert_user(username: str, password: str, role: str = "admin") -> AdminUser:
    username = (username or "").strip()
    if not username:
        raise ValueError("username required")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if is_weak_password(password):
        raise ValueError("password too weak: use at least 10 characters and avoid common passwords")
    store = _load_store()
    users = [u for u in store.get("users", []) if isinstance(u, dict) and u.get("username") != username]
    users.append({"username": username, "role": role, "hash": hash_password(password)})
    store["users"] = users
    _save_store(store)
    return AdminUser(username, role, "")


def remove_user(username: str) -> bool:
    store = _load_store()
    before = len(store.get("users", []))
    store["users"] = [u for u in store.get("users", []) if not (isinstance(u, dict) and u.get("username") == username)]
    _save_store(store)
    return len(store["users"]) < before


def _env_fallback_user() -> Optional[AdminUser]:
    """Single account from env. Refuses weak/default creds in HIPAA mode."""
    user = (os.getenv("ADMIN_USER") or "").strip()
    pw_hash = (os.getenv("ADMIN_PASS_HASH") or "").strip()
    pw = os.getenv("ADMIN_PASS")
    if not user:
        return None
    if pw_hash:
        return AdminUser(user, "admin", pw_hash)
    if pw is None:
        return None
    if sc.hipaa_mode() and is_weak_password(pw):
        return None
    return AdminUser(user, "admin", hash_password(pw))


def find_user(username: str) -> Optional[AdminUser]:
    username = (username or "").strip()
    for u in list_users():
        if u.username == username:
            return u
    env_user = _env_fallback_user()
    if env_user and env_user.username == username:
        return env_user
    return None


def admin_configured() -> bool:
    return bool(list_users()) or _env_fallback_user() is not None


def configuration_problem() -> str:
    """Why admin login is unavailable (empty string when fine)."""
    if admin_configured():
        return ""
    if (os.getenv("ADMIN_PASS") or "") and sc.hipaa_mode():
        return (
            "ADMIN_PASS is too weak or a known default (e.g. 12345); admin login is disabled. "
            "Create an account: python manage_admin.py add <username>"
        )
    return "No admin account configured. Create one: python manage_admin.py add <username>"


# ------------------------------------------------------------- sessions ----
@dataclass
class Session:
    sid: str
    username: str
    role: str
    created: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)


_SESSIONS: dict[str, Session] = {}
_SLOCK = threading.Lock()


def _sign(sid: str) -> str:
    return hmac.new(sc.server_secret().encode(), sid.encode(), hashlib.sha256).hexdigest()[:40]


def create_session(user: AdminUser) -> str:
    sid = secrets.token_urlsafe(24)
    with _SLOCK:
        _SESSIONS[sid] = Session(sid=sid, username=user.username, role=user.role)
    return f"{sid}.{_sign(sid)}"


def resolve_session(token: str | None) -> Optional[Session]:
    if not token or "." not in token:
        return None
    sid, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(sid)):
        return None
    now = time.monotonic()
    with _SLOCK:
        sess = _SESSIONS.get(sid)
        if not sess:
            return None
        if now - sess.created > ABSOLUTE_TIMEOUT_SECS or now - sess.last_seen > IDLE_TIMEOUT_SECS:
            _SESSIONS.pop(sid, None)
            return None
        sess.last_seen = now
        return sess


def revoke_session(token: str | None) -> None:
    if not token or "." not in token:
        return
    sid = token.rsplit(".", 1)[0]
    with _SLOCK:
        _SESSIONS.pop(sid, None)


def revoke_all_sessions() -> int:
    with _SLOCK:
        n = len(_SESSIONS)
        _SESSIONS.clear()
    return n


# ---------------------------------------------------------------- login ----
class LoginError(Exception):
    def __init__(self, message: str, status: int = 401, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def login(username: str, password: str, *, client_key: str = "") -> tuple[AdminUser, str]:
    """Validate credentials; returns (user, session token). Raises LoginError."""
    username = (username or "").strip()
    keys = [f"u:{username}"]
    if client_key:
        keys.append(f"ip:{client_key}")
        if not _LOGIN_RATE.allow(f"ip:{client_key}"):
            raise LoginError("Too many login attempts. Try again shortly.", 429, _LOGIN_RATE.retry_after(f"ip:{client_key}"))
    for k in keys:
        if _LOCKOUT.is_locked(k):
            raise LoginError("Account temporarily locked after repeated failures.", 423, _LOCKOUT.lock_secs)
    problem = configuration_problem()
    if problem:
        raise LoginError(problem, 503)
    user = find_user(username)
    ok = bool(user) and verify_password(password or "", user.hash)
    if not ok:
        # constant-ish time: hash anyway when user missing
        if not user:
            verify_password(password or "", hash_password("x" * 12))
        for k in keys:
            _LOCKOUT.record_failure(k)
        raise LoginError("Invalid username or password", 401)
    for k in keys:
        _LOCKOUT.record_success(k)
    return user, create_session(user)  # type: ignore[arg-type]


def cookie_kwargs() -> dict[str, Any]:
    return {
        "httponly": True,
        "samesite": "strict",
        "secure": sc.cookie_secure(),
        "max_age": ABSOLUTE_TIMEOUT_SECS,
        "path": "/",
    }
