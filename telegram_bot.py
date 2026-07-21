"""Telegram channel for hospital appointment booking.

Uses the same Main.py LangGraph + Tools.py flow as CLI / voice Cascade.
Requires TELEGRAM_BOT_TOKEN in `.env`.

    python telegram_bot.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from agent_turn import run_turn, sanitize_assistant_reply  # noqa: E402
from conversation_log import (  # noqa: E402
    append_or_update_turn,
    end_call,
    set_current_call_id,
    start_call,
)
from conversation_log import _utc_iso  # noqa: E402
from voice_bridge import get_shared_hospital_graph  # noqa: E402

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE_LEN = 4000

_WELCOME = (
    "Hi — I'm the DBC Care front desk.\n\n"
    "Tell me what brings you in today (a visit, medicines, or anything else we can help with). "
    "I'll ask for your phone when we need to pull up your record.\n\n"
    "Commands:\n"
    "/start — welcome\n"
    "/help — how this works\n"
    "/reset — start a fresh conversation"
)

_HELP = (
    "Just say what you need in your own words — no menus.\n"
    "I'll greet you, learn what you need, then ask for phone and name to look you up.\n"
    "Clinic hours: 9:00 AM–5:00 PM (lunch 2:00–3:00 PM).\n"
    "Use /reset if you want to start over."
)


class TelegramSession:
    """Per-chat LangGraph thread + admin chat log."""

    def __init__(self, chat_id: int, user_id: str = ""):
        self.chat_id = chat_id
        self.user_id = user_id or f"tg_{chat_id}"
        self.thread_id = f"tg_{chat_id}"
        self.call_id = f"tg_{chat_id}_{uuid.uuid4().hex[:8]}"
        self._started = False

    def ensure_started(self) -> None:
        if self._started:
            return
        start_call(
            self.call_id,
            pipeline_mode="telegram",
            session_id=self.thread_id,
            channel="telegram",
            user_id=self.user_id,
            audio_codec="none",
            extra={"telegram_chat_id": self.chat_id},
        )
        self._started = True

    def reset(self) -> None:
        if self._started:
            end_call(self.call_id)
        self.thread_id = f"tg_{self.chat_id}_{uuid.uuid4().hex[:6]}"
        self.call_id = f"tg_{self.chat_id}_{uuid.uuid4().hex[:8]}"
        self._started = False


class TelegramBot:
    def __init__(self, token: str):
        self.token = token.strip()
        self.base = f"{TELEGRAM_API}/bot{self.token}"
        self.sessions: dict[int, TelegramSession] = {}
        self.graph = get_shared_hospital_graph()
        self._offset = 0

    def _session(self, chat_id: int, from_user: dict[str, Any] | None = None) -> TelegramSession:
        sess = self.sessions.get(chat_id)
        if sess is None:
            uid = ""
            if from_user:
                uid = f"tg_{from_user.get('id') or chat_id}"
            sess = TelegramSession(chat_id, user_id=uid)
            self.sessions[chat_id] = sess
        return sess

    async def _api(self, client: httpx.AsyncClient, method: str, **payload: Any) -> dict[str, Any]:
        r = await client.post(f"{self.base}/{method}", json=payload, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error ({method}): {data}")
        return data.get("result")

    async def get_me(self, client: httpx.AsyncClient) -> dict[str, Any]:
        return await self._api(client, "getMe")

    async def send_chat_action(self, client: httpx.AsyncClient, chat_id: int, action: str = "typing") -> None:
        try:
            await self._api(client, "sendChatAction", chat_id=chat_id, action=action)
        except Exception:
            pass

    async def send_message(self, client: httpx.AsyncClient, chat_id: int, text: str) -> None:
        body = (text or "").strip() or "…"
        for chunk in _chunk_text(body, MAX_MESSAGE_LEN):
            await self._api(
                client,
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                disable_web_page_preview=True,
            )

    async def get_updates(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        # Long poll — Telegram holds until timeout or updates arrive.
        r = await client.get(
            f"{self.base}/getUpdates",
            params={"offset": self._offset, "timeout": 30},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"getUpdates failed: {data}")
        return list(data.get("result") or [])

    def _logged_turn(self, sess: TelegramSession, user_text: str) -> str:
        sess.ensure_started()
        set_current_call_id(sess.call_id)
        user_sent = _utc_iso()
        append_or_update_turn(
            sess.call_id,
            {
                "mode": "text",
                "input_type": "text",
                "user_text": user_text,
                "user_sent_at": user_sent,
                "bot_received_at": user_sent,
            },
            new_turn=True,
        )
        text_start = _utc_iso()
        turn = run_turn(self.graph, user_text, sess.thread_id, call_id=sess.call_id)
        reply = sanitize_assistant_reply(getattr(turn, "text", None) or str(turn or ""))
        if not reply:
            reply = "Sorry, I couldn't process that. Could you try again?"
        agent_name = getattr(turn, "agent", "") or ""
        text_end = _utc_iso()
        append_or_update_turn(
            sess.call_id,
            {
                "bot_text": reply,
                "agent_name": agent_name,
                "bot_text_first_token_at": text_start,
                "bot_text_first_shown_at": text_start,
                "bot_text_complete_at": text_end,
            },
            new_turn=False,
        )
        return reply

    async def handle_message(self, client: httpx.AsyncClient, message: dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        chat_id = int(chat_id)
        text = (message.get("text") or "").strip()
        if not text:
            await self.send_message(
                client,
                chat_id,
                "Please send a text message — tell me what brings you in today.",
            )
            return

        sess = self._session(chat_id, message.get("from"))
        cmd = text.split()[0].lower().split("@", 1)[0]

        if cmd in ("/start", "/help"):
            await self.send_message(client, chat_id, _WELCOME if cmd == "/start" else _HELP)
            return

        if cmd == "/reset":
            sess.reset()
            self.sessions[chat_id] = sess
            await self.send_message(
                client,
                chat_id,
                "Conversation cleared. Hi — what brings you in today?",
            )
            return

        await self.send_chat_action(client, chat_id, "typing")
        try:
            reply = await asyncio.to_thread(self._logged_turn, sess, text)
        except Exception:
            logger.exception("Telegram turn failed chat_id={}", chat_id)
            reply = "Something went wrong on my side. Please try again in a moment."
        await self.send_message(client, chat_id, reply)

    async def process_update(self, client: httpx.AsyncClient, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            self._offset = max(self._offset, update_id + 1)
        message = update.get("message") or update.get("edited_message")
        if message:
            await self.handle_message(client, message)

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            # Clear any webhook so long-polling works.
            try:
                await self._api(client, "deleteWebhook", drop_pending_updates=False)
            except Exception:
                logger.warning("deleteWebhook failed; continuing with getUpdates")
            me = await self.get_me(client)
            username = me.get("username") or me.get("first_name") or "bot"
            logger.info("Telegram bot online as @{} (id={})", username, me.get("id"))
            print(f"Telegram bot ready: @{username}")
            print("Open Telegram, message the bot, and book an appointment.")
            print("Ctrl+C to stop.\n")
            while True:
                try:
                    updates = await self.get_updates(client)
                except httpx.TimeoutException:
                    continue
                except Exception:
                    logger.exception("getUpdates error; retrying in 3s")
                    await asyncio.sleep(3)
                    continue
                for update in updates:
                    try:
                        await self.process_update(client, update)
                    except Exception:
                        logger.exception("Failed to process update {}", update.get("update_id"))


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return chunks


def main() -> None:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is missing. Add it to .env, then run: python telegram_bot.py"
        )
    bot = TelegramBot(token)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nTelegram bot stopped.")
        for sess in bot.sessions.values():
            if sess._started:
                end_call(sess.call_id)


if __name__ == "__main__":
    main()
