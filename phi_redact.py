"""PHI minimisation helpers for logs and operational records.

* ``mask_phone`` / ``mask_name`` / ``redact_text`` — irreversible masking.
* ``redact_obj`` — walk dict/list payloads (tool args/results) and mask
  identifying fields while keeping operational keys (ids, ok, counts).
* ``install_log_redaction`` — attach a filter to stdlib ``logging`` and a
  patcher to loguru so any phone-like digit run or e-mail in a log line is
  masked before it reaches ``app.log``. Safety net only — call sites should
  still avoid logging PHI.
"""

from __future__ import annotations

import logging
import re
from typing import Any

# 7+ digit runs, allowing spaces/dashes/dots/parentheses between groups, optional +CC
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{6,}\d)(?!\w)")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Keys whose values are direct identifiers (case-insensitive match on key name).
_IDENTIFIER_KEYS = {
    "phone", "phone_digits", "confirm_phone_from_db", "patient_phone",
    "name", "patient_name", "confirm_name_from_db", "full_name",
    "address", "email",
    "user_text", "bot_text", "user_input_text", "bot_response_text", "text",
    "medicine_name", "timing", "prescriptions",
}
_KEEP_KEYS = {"doctor", "doctor_name", "department", "status", "ok", "message"}


def mask_phone(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return ""
    if len(digits) <= 2:
        return "*" * len(digits)
    return "*" * (len(digits) - 2) + digits[-2:]


def mask_name(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    parts = s.split()
    return " ".join((p[0] + "*" * (len(p) - 1)) if p else p for p in parts)


def redact_text(text: Any) -> str:
    s = str(text or "")
    if not s:
        return s
    s = _EMAIL_RE.sub("[email]", s)
    s = _PHONE_RE.sub(lambda m: "[phone:" + mask_phone(m.group(1))[-2:] + "]", s)
    return s


def _is_identifier_key(key: str) -> bool:
    k = (key or "").lower()
    if k in _KEEP_KEYS:
        return False
    return k in _IDENTIFIER_KEYS


def redact_obj(obj: Any, *, depth: int = 0) -> Any:
    """Return a copy of *obj* with identifying values masked."""
    if depth > 12:
        return "[depth]"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if _is_identifier_key(ks):
                if "phone" in ks.lower():
                    out[ks] = mask_phone(v)
                elif ks.lower() in {"name", "patient_name", "confirm_name_from_db", "full_name"}:
                    out[ks] = mask_name(v)
                elif isinstance(v, (dict, list)):
                    out[ks] = redact_obj(v, depth=depth + 1)
                else:
                    out[ks] = "[redacted]" if v not in (None, "", [], {}) else v
            else:
                out[ks] = redact_obj(v, depth=depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_obj(v, depth=depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact_text(obj)
    return obj


def summarize_messages(messages: Any) -> str:
    """Non-PHI description of a LangGraph message list for debug logs."""
    try:
        msgs = list(messages or [])
    except TypeError:
        return "messages=?"
    roles: list[str] = []
    for m in msgs[-6:]:
        role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None) or "?"
        roles.append(str(role))
    return f"messages={len(msgs)} tail={roles}"


# ------------------------------------------------------------ log hooks ----
class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        red = redact_text(msg)
        if red != msg:
            record.msg = red
            record.args = ()
        return True


_INSTALLED = False


def install_log_redaction() -> None:
    """Idempotent: redact stdlib logging root handlers + loguru records."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    flt = _RedactingFilter()
    root = logging.getLogger()
    root.addFilter(flt)
    for h in list(root.handlers):
        h.addFilter(flt)
    try:
        from loguru import logger as _loguru

        def _patch(record: dict) -> None:
            try:
                record["message"] = redact_text(record["message"])
            except Exception:  # noqa: BLE001
                pass

        _loguru.configure(patcher=_patch)
    except Exception:  # noqa: BLE001
        pass
