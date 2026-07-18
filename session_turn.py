"""Latest user utterance for the active pipeline turn (RAG logging, etc.)."""

from __future__ import annotations

_last_user_message: str = ""


def set_last_user_message(text: str) -> None:
    global _last_user_message
    cleaned = (text or "").strip()
    if cleaned:
        _last_user_message = cleaned[:500]


def get_last_user_message() -> str:
    return _last_user_message


def clear_last_user_message() -> None:
    global _last_user_message
    _last_user_message = ""
