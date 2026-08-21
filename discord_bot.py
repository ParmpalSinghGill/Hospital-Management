"""Discord channel for hospital appointment booking.

Uses the same Main.py LangGraph + Tools.py flow as Telegram / CLI / voice.
Requires DISCORD_BOT_TOKEN in `.env`.

Works in DMs and when @mentioned in a server (no privileged Message Content
Intent required for those cases). Optionally enable Message Content Intent in
the Developer Portal if you want the bot to read every guild message.

    python discord_bot.py
"""

from __future__ import annotations

import asyncio
import atexit
import fcntl
import os
import sys
import uuid
from pathlib import Path

import discord
from dotenv import load_dotenv
from loguru import logger

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

_LOCK_PATH = _ROOT / ".discord_bot.lock"
_lock_fp = None


def _acquire_singleton_lock() -> None:
    """Exit if another discord_bot.py is already running (avoids double replies)."""
    global _lock_fp
    _lock_fp = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _lock_fp.close()
        _lock_fp = None
        raise SystemExit(
            "Another discord_bot.py is already running. Stop it first "
            "(pkill -f 'python discord_bot.py'), then start only one instance."
        )
    _lock_fp.write(str(os.getpid()))
    _lock_fp.flush()

    def _release() -> None:
        global _lock_fp
        if _lock_fp is None:
            return
        try:
            fcntl.flock(_lock_fp.fileno(), fcntl.LOCK_UN)
            _lock_fp.close()
        except Exception:
            pass
        _lock_fp = None

    atexit.register(_release)

from agent_turn import run_turn, sanitize_assistant_reply  # noqa: E402
from conversation_log import (  # noqa: E402
    append_or_update_turn,
    end_call,
    set_current_call_id,
    start_call,
)
from conversation_log import _utc_iso  # noqa: E402
from voice_bridge import get_shared_hospital_graph  # noqa: E402

MAX_MESSAGE_LEN = 1900  # Discord limit is 2000; leave headroom

_WELCOME = (
    "Hi — I'm the DBC Care front desk.\n\n"
    "Tell me what brings you in today (a visit, medicines, or anything else we can help with). "
    "I'll ask for your phone when we need to pull up your record.\n\n"
    "Commands:\n"
    "`!start` or `/start` — welcome\n"
    "`!help` or `/help` — how this works\n"
    "`!reset` or `/reset` — start a fresh conversation"
)

_HELP = (
    "Just say what you need in your own words — no menus.\n"
    "I'll greet you, learn what you need, then ask for phone and name to look you up.\n"
    "Clinic hours: 9:00 AM–5:00 PM (lunch 2:00–3:00 PM).\n"
    "Prefer DMs for privacy. Use `!reset` if you want to start over."
)


class DiscordSession:
    """Per-user LangGraph thread + admin chat log."""

    def __init__(self, user_id: int, channel_id: int):
        self.user_id = user_id
        self.channel_id = channel_id
        self.thread_id = f"dc_{user_id}"
        self.call_id = f"dc_{user_id}_{uuid.uuid4().hex[:8]}"
        self.log_user_id = f"dc_{user_id}"
        self._started = False

    def ensure_started(self) -> None:
        if self._started:
            return
        start_call(
            self.call_id,
            pipeline_mode="discord",
            session_id=self.thread_id,
            channel="discord",
            user_id=self.log_user_id,
            audio_codec="none",
            extra={"discord_user_id": self.user_id, "discord_channel_id": self.channel_id},
        )
        self._started = True

    def reset(self) -> None:
        if self._started:
            end_call(self.call_id)
        self.thread_id = f"dc_{self.user_id}_{uuid.uuid4().hex[:6]}"
        self.call_id = f"dc_{self.user_id}_{uuid.uuid4().hex[:8]}"
        self._started = False


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


def _normalize_command(text: str) -> str:
    """Return slash-style command name if message is a command, else ''."""
    raw = (text or "").strip()
    if not raw:
        return ""
    first = raw.split()[0].lower()
    if first.startswith("/"):
        return first.split("@", 1)[0]
    if first.startswith("!"):
        return "/" + first[1:].split("@", 1)[0]
    return ""


class HospitalDiscordBot(discord.Client):
    def __init__(self) -> None:
        # Default intents are enough for DMs + @mentions (message content is
        # delivered for those without the privileged Message Content Intent).
        intents = discord.Intents.default()
        intents.dm_messages = True
        # Optional: set DISCORD_MESSAGE_CONTENT_INTENT=1 after enabling the
        # privileged intent in the Developer Portal (reads all guild messages).
        if (os.getenv("DISCORD_MESSAGE_CONTENT_INTENT") or "").strip() in ("1", "true", "yes"):
            intents.message_content = True
        super().__init__(intents=intents)
        self.sessions: dict[int, DiscordSession] = {}
        self.graph = get_shared_hospital_graph()

    def _session(self, user_id: int, channel_id: int) -> DiscordSession:
        sess = self.sessions.get(user_id)
        if sess is None:
            sess = DiscordSession(user_id, channel_id)
            self.sessions[user_id] = sess
        else:
            sess.channel_id = channel_id
        return sess

    async def on_ready(self) -> None:
        user = self.user
        name = user.name if user else "bot"
        logger.info("Discord bot online as {} (id={})", name, user.id if user else "?")
        print(f"Discord bot ready: {name}")
        print("DM the bot (or mention it in a server channel) to book an appointment.")
        print("Ctrl+C to stop.\n")

    def _logged_turn(self, sess: DiscordSession, user_text: str) -> str:
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

    async def _send_chunks(self, channel: discord.abc.Messageable, text: str) -> None:
        body = (text or "").strip() or "…"
        for chunk in _chunk_text(body, MAX_MESSAGE_LEN):
            await channel.send(chunk)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if self.user is None:
            return

        # In servers: only reply when mentioned or in a DM.
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentioned = self.user in message.mentions
        if not is_dm and not mentioned:
            return

        text = (message.content or "").strip()
        if mentioned and self.user:
            text = text.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "").strip()

        if not text:
            await message.channel.send(
                "Please send a text message — tell me what brings you in today."
            )
            return

        sess = self._session(message.author.id, message.channel.id)
        cmd = _normalize_command(text)

        if cmd in ("/start", "/help"):
            await self._send_chunks(message.channel, _WELCOME if cmd == "/start" else _HELP)
            return

        if cmd == "/reset":
            sess.reset()
            self.sessions[message.author.id] = sess
            await message.channel.send("Conversation cleared. Hi — what brings you in today?")
            return

        async with message.channel.typing():
            try:
                reply = await asyncio.to_thread(self._logged_turn, sess, text)
            except Exception:
                logger.exception("Discord turn failed user_id={}", message.author.id)
                reply = "Something went wrong on my side. Please try again in a moment."
        await self._send_chunks(message.channel, reply)


def main() -> None:
    _acquire_singleton_lock()
    token = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is missing. Add it to .env, then run: python discord_bot.py"
        )
    bot = HospitalDiscordBot()
    try:
        bot.run(token, log_handler=None)
    except KeyboardInterrupt:
        print("\nDiscord bot stopped.")
    finally:
        for sess in bot.sessions.values():
            if sess._started:
                end_call(sess.call_id)


if __name__ == "__main__":
    main()
