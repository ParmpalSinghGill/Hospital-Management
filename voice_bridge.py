"""Bridge: existing Main.py + Tools.py ↔ Pipecat voice pipelines.

Cascade: Deepgram STT → this LangGraph processor → Deepgram TTS
Realtime: OpenAI Realtime function-calls → Tools.py (same booking tools)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_turn import run_turn, sanitize_assistant_reply  # noqa: E402
from Main import build_graph  # noqa: E402
from Tools import (  # noqa: E402
    book_appointment,
    cancel_appointment,
    get_prescriptions,
    list_doctors,
    lookup_patient,
    reschedule_appointment,
    save_patient,
)

# LangChain @tool objects from the *existing* Tools.py
_LC_TOOLS = {
    "list_doctors": list_doctors,
    "lookup_patient": lookup_patient,
    "save_patient": save_patient,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "reschedule_appointment": reschedule_appointment,
    "get_prescriptions": get_prescriptions,
}

# One shared LangGraph + checkpointer for the whole bot process so the same
# browser conversation (thread_id) keeps memory across call reconnects.
_SHARED_GRAPH = None
_SHARED_GRAPH_PROVIDER: str | None = None


def create_hospital_graph(*, llm_provider: str | None = None):
    """Build the same LangGraph as CLI, with Cascade using admin/cascade LLM."""
    import os

    from service_settings import apply_settings_to_env, load_settings

    apply_settings_to_env()
    settings = load_settings()
    provider = llm_provider or settings.get("cascade_llm") or "deepseek"
    os.environ["LLM_PROVIDER"] = provider
    # Agents call _init_model with CLI_LLM_PROVIDER first — override for this graph
    os.environ["CLI_LLM_PROVIDER"] = provider
    return build_graph()


def get_shared_hospital_graph(*, llm_provider: str | None = None):
    """Reuse one compiled graph so InMemorySaver checkpoints survive new calls."""
    global _SHARED_GRAPH, _SHARED_GRAPH_PROVIDER
    import os

    from service_settings import apply_settings_to_env, load_settings

    apply_settings_to_env()
    settings = load_settings()
    provider = llm_provider or settings.get("cascade_llm") or "deepseek"
    if _SHARED_GRAPH is None or _SHARED_GRAPH_PROVIDER != provider:
        logger.info(f"Building shared hospital graph (provider={provider})")
        _SHARED_GRAPH = create_hospital_graph(llm_provider=provider)
        _SHARED_GRAPH_PROVIDER = provider
    else:
        os.environ["LLM_PROVIDER"] = provider
        os.environ["CLI_LLM_PROVIDER"] = provider
    return _SHARED_GRAPH


def _last_user_text(context) -> str:
    """Best-effort extract of the latest user utterance from an LLMContext."""
    messages = getattr(context, "messages", None) or []
    for msg in reversed(list(messages)):
        role = getattr(msg, "role", None) or getattr(msg, "type", None)
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            content = msg.get("content", "")
        else:
            content = getattr(msg, "content", "")
        role_s = str(role or "").lower()
        if role_s in ("user", "human"):
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    else:
                        parts.append(str(getattr(p, "text", p)))
                return " ".join(parts).strip()
            return str(content).strip()
    return ""


# --- Phone-fragment merge (STT text only; does not touch VAD / mute) ---

_PHONE_FRAGMENT_WAIT_S = 1.15

_SPOKEN_DIGIT = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_PHONE_FILLER = frozenset(
    {
        "a",
        "and",
        "double",
        "full",
        "is",
        "it",
        "it's",
        "its",
        "my",
        "number",
        "phone",
        "please",
        "so",
        "that's",
        "thats",
        "the",
        "triple",
        "uh",
        "um",
        "yes",
    }
)


def merge_utterances(left: str, right: str) -> str:
    """Join two utterances; drop exact duplicates / containment."""
    a = (left or "").strip()
    b = (right or "").strip()
    if not a:
        return b
    if not b:
        return a
    if b == a or b in a:
        return a
    if a in b:
        return b
    return f"{a} {b}"


def transcript_phone_digits(text: str) -> str:
    """Pull digit characters from STT text (spoken words + numerals)."""
    import re

    raw = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9']+", raw)
    digits: list[str] = []
    repeat = 1
    for tok in tokens:
        if tok in ("double", "triple"):
            repeat = 2 if tok == "double" else 3
            continue
        if tok.isdigit():
            digits.append(tok)
            repeat = 1
            continue
        mapped = _SPOKEN_DIGIT.get(tok)
        if mapped is not None:
            digits.append(mapped * repeat)
            repeat = 1
            continue
        repeat = 1
    return "".join(digits)


def bot_awaiting_phone(bot_text: str) -> bool:
    """True when the last bot reply was asking for a phone / number."""
    t = str(bot_text or "").lower()
    return any(
        key in t
        for key in (
            "phone",
            "mobile",
            "number",
            "digit",
            "contact",
        )
    )


def looks_like_phone_fragment(text: str) -> bool:
    """True if STT text is mostly an incomplete phone (digit words / numerals).

    Complete 10+ digit numbers return False so they go to the agent immediately.
    """
    import re

    from db_patients import _phone_is_complete

    raw = str(text or "").strip()
    if not raw:
        return False
    digits = transcript_phone_digits(raw)
    if len(digits) < 2:
        return False
    if _phone_is_complete(digits):
        return False

    tokens = re.findall(r"[a-z0-9']+", raw.lower())
    leftover = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("double", "triple"):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if nxt in _SPOKEN_DIGIT or (nxt.isdigit() and len(nxt) == 1):
                i += 2
                continue
            leftover.append(tok)
            i += 1
            continue
        if tok in _SPOKEN_DIGIT or tok.isdigit() or tok in _PHONE_FILLER:
            i += 1
            continue
        leftover.append(tok)
        i += 1
    # Allow a couple of soft words ("it's", already filler) — reject real sentences.
    return len(leftover) == 0


def make_langgraph_processor(
    thread_id: str,
    graph_app: Any,
    *,
    call_id: str | None = None,
    shared_state: dict[str, Any] | None = None,
):
    """Pipecat FrameProcessor that runs Main.py LangGraph each user turn."""
    import time
    from dataclasses import dataclass

    from pipecat.frames.frames import (
        Frame,
        LLMContextFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        OutputTransportMessageUrgentFrame,
        TTSSpeakFrame,
        UninterruptibleFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    log_call_id = call_id or thread_id
    state = shared_state if shared_state is not None else {}

    # Keep reply text/audio in the queue if an InterruptionFrame arrives mid-push.
    @dataclass
    class GuardedLLMFullResponseStartFrame(LLMFullResponseStartFrame, UninterruptibleFrame):
        pass

    @dataclass
    class GuardedLLMFullResponseEndFrame(LLMFullResponseEndFrame, UninterruptibleFrame):
        pass

    @dataclass
    class GuardedTTSSpeakFrame(TTSSpeakFrame, UninterruptibleFrame):
        pass

    class HospitalLangGraphProcessor(FrameProcessor):
        def __init__(self):
            super().__init__()
            self._busy = False
            self._last_user_text = ""
            self._last_user_at = 0.0
            self._pending_user_text = ""
            self._phone_buffer = ""
            self._phone_flush_task: asyncio.Task | None = None

        def _cancel_phone_flush(self) -> None:
            task = self._phone_flush_task
            self._phone_flush_task = None
            if task and not task.done():
                task.cancel()

        def _queue_while_busy(self, user_text: str) -> None:
            """Append (don't overwrite) so digit fragments aren't dropped."""
            self._pending_user_text = merge_utterances(self._pending_user_text, user_text)
            logger.info(f"Queuing user turn while busy: {user_text!r}")
            try:
                from conversation_log import record_timeline_event

                record_timeline_event(
                    log_call_id,
                    event_type="queued_while_busy",
                    user_text=user_text,
                    bot_text=str(state.get("last_bot_text") or ""),
                    phase="agent_busy",
                )
            except Exception:
                pass

        def _start_turn(self, user_text: str, direction: FrameDirection) -> None:
            self._busy = True
            state["busy"] = True
            self.create_task(
                self._drain_turns(user_text, direction),
                name="hospital_langgraph_turn",
            )

        async def _flush_phone_buffer(self, direction: FrameDirection) -> None:
            """After a short pause, send buffered digit fragments as one turn."""
            try:
                await asyncio.sleep(_PHONE_FRAGMENT_WAIT_S)
            except asyncio.CancelledError:
                return
            self._phone_flush_task = None
            text = (self._phone_buffer or "").strip()
            self._phone_buffer = ""
            if not text:
                return
            logger.info(f"Flushing phone fragment buffer: {text!r}")
            if self._busy:
                self._queue_while_busy(text)
                return
            self._start_turn(text, direction)

        def _buffer_phone_fragment(self, user_text: str, direction: FrameDirection) -> None:
            from db_patients import _phone_is_complete

            self._phone_buffer = merge_utterances(self._phone_buffer, user_text)
            digits = transcript_phone_digits(self._phone_buffer)
            logger.info(
                f"Buffering phone fragment ({len(digits)} digits): {user_text!r} → {self._phone_buffer!r}"
            )
            if _phone_is_complete(digits):
                self._cancel_phone_flush()
                text = self._phone_buffer.strip()
                self._phone_buffer = ""
                if self._busy:
                    self._queue_while_busy(text)
                else:
                    self._start_turn(text, direction)
                return
            self._cancel_phone_flush()
            self._phone_flush_task = self.create_task(
                self._flush_phone_buffer(direction),
                name="hospital_phone_fragment_flush",
            )

        async def _push_spoken_reply(self, reply: str, direction: FrameDirection):
            """Show full text in chat, synthesize as one TTS utterance.

            Default sentence-aggregation TTS splits multi-sentence replies into
            separate audio jobs. Gaps between those jobs briefly unmute the mic,
            and speaker echo often interrupts — so chat shows the full message
            while audio cuts halfway. One TTSSpeakFrame avoids that gap.

            Do NOT fake BotStartedSpeakingFrame here: if TTS is then interrupted
            before BotStoppedSpeakingFrame, AlwaysUserMute-style strategies leave
            the mic muted forever (pin-drop silence).
            """
            state["last_bot_text"] = reply
            start = GuardedLLMFullResponseStartFrame()
            start.skip_tts = True
            end = GuardedLLMFullResponseEndFrame()
            end.skip_tts = True
            # Chat bubble via one channel only (server-message). Do NOT also push
            # LLMTextFrame — that duplicated every reply in the UI (LLM text +
            # server-message). Start/End still fire for turn metrics / RTVI.
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={
                        "label": "rtvi-ai",
                        "type": "server-message",
                        "data": {"type": "bot_chat_text", "text": reply},
                    }
                ),
                direction,
            )
            await self.push_frame(start, direction)
            await self.push_frame(end, direction)
            await self.push_frame(
                GuardedTTSSpeakFrame(text=reply, append_to_context=False),
                direction,
            )

        async def _run_one_turn(self, user_text: str, direction: FrameDirection) -> None:
            now = time.monotonic()
            self._last_user_text = user_text
            self._last_user_at = now
            state["last_user_text"] = user_text
            state["busy"] = True
            logger.info(f"LangGraph turn ({thread_id}): {user_text!r}")
            # Run inference off the process-frame task. If we await here inside
            # process_frame and swallow CancelledError, Pipecat's barge-in waits
            # forever for the process task to exit → dead silence.
            turn_result = await asyncio.to_thread(
                run_turn,
                graph_app,
                user_text,
                thread_id,
                call_id=log_call_id,
            )
            reply = getattr(turn_result, "text", None) or str(turn_result or "")
            agent = getattr(turn_result, "agent", "") or ""
            reply = sanitize_assistant_reply(reply or "")
            if not reply:
                reply = "Sorry, I couldn't process that. Could you say it again?"
            if agent:
                logger.info(f"LangGraph reply via [{agent}]: {reply!r}")
            else:
                logger.info(f"LangGraph reply: {reply!r}")
            await self._push_spoken_reply(reply, direction)

        async def _drain_turns(self, first_text: str, direction: FrameDirection) -> None:
            """Background turn loop — must not run on Pipecat's process-frame task."""
            try:
                next_text = first_text
                while next_text:
                    try:
                        await self._run_one_turn(next_text, direction)
                    except Exception as e:
                        logger.exception("LangGraph turn failed")
                        err = f"Sorry, something went wrong: {e}"
                        try:
                            await self._push_spoken_reply(err, direction)
                        except Exception:
                            logger.exception("Failed to speak LangGraph error")
                    pending = (self._pending_user_text or "").strip()
                    self._pending_user_text = ""
                    if not pending or pending == next_text:
                        break
                    if (
                        pending == self._last_user_text
                        and (time.monotonic() - self._last_user_at) < 2.0
                    ):
                        break
                    next_text = pending
            finally:
                self._busy = False
                state["busy"] = False

        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)

            if isinstance(frame, LLMContextFrame):
                user_text = _last_user_text(frame.context)
                if not user_text:
                    # Do not block waiting for STT — that open cancel window
                    # was dropping later turns and killing voice.
                    await self.push_frame(frame, direction)
                    return

                now = time.monotonic()
                if (
                    user_text == self._last_user_text
                    and (now - self._last_user_at) < 2.0
                ):
                    logger.debug(f"Skipping duplicate LangGraph turn: {user_text!r}")
                    return

                awaiting = bot_awaiting_phone(str(state.get("last_bot_text") or ""))
                if awaiting and looks_like_phone_fragment(user_text):
                    if self._busy:
                        self._queue_while_busy(user_text)
                    else:
                        self._buffer_phone_fragment(user_text, direction)
                    return

                # Non-fragment while digits were buffering — flush together once.
                if self._phone_buffer:
                    self._cancel_phone_flush()
                    user_text = merge_utterances(self._phone_buffer, user_text)
                    self._phone_buffer = ""
                    logger.info(f"Flushing phone buffer with follow-up: {user_text!r}")

                if self._busy:
                    self._queue_while_busy(user_text)
                    return

                # Return immediately so barge-in can cancel/recreate the process
                # task while LangGraph keeps running in the background.
                self._start_turn(user_text, direction)
                return

            await self.push_frame(frame, direction)

    return HospitalLangGraphProcessor()


def hospital_function_schemas():
    """Pipecat FunctionSchema list mirroring Tools.py."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema

    return [
        FunctionSchema(
            name="list_doctors",
            description=(
                getattr(list_doctors, "description", None)
                or "List doctors, optionally filtered by department or name."
            ),
            properties={
                "department": {"type": "string", "description": "Department filter."},
                "query": {"type": "string", "description": "Doctor name filter."},
                "exclude_doctor": {
                    "type": "string",
                    "description": "Omit this doctor (e.g. previous doctor for a second opinion).",
                },
                "limit": {"type": "integer", "description": "Max results (default 10)."},
            },
            required=[],
        ),
        FunctionSchema(
            name="lookup_patient",
            description=(
                getattr(lookup_patient, "description", None)
                or "Look up patient by phone (and name to verify). Returns past doctors for returning patients."
            ),
            properties={
                "phone": {"type": "string", "description": "Patient phone number."},
                "patient_name": {"type": "string", "description": "Full name for verification."},
                "patient_id": {"type": "string", "description": "Optional id like PAT-0001."},
                "department": {
                    "type": "string",
                    "description": (
                        "Optional specialty after symptoms are known. Returns booking_guidance: "
                        "inform_existing_soon, offer_prepone, or offer_new_booking."
                    ),
                },
            },
            required=[],
        ),
        FunctionSchema(
            name="save_patient",
            description=(
                getattr(save_patient, "description", None)
                or "Create or update patient after phone and name are confirmed. Writes to patients table."
            ),
            properties={
                "patient_name": {"type": "string", "description": "Confirmed full name."},
                "phone": {"type": "string", "description": "Confirmed phone number."},
                "address": {"type": "string", "description": "Optional address."},
            },
            required=["patient_name", "phone"],
        ),
        FunctionSchema(
            name="book_appointment",
            description=(
                getattr(book_appointment, "description", None)
                or "Book an appointment (patient_name, phone, doctor, time; address optional)."
            ),
            properties={
                "patient_name": {"type": "string"},
                "phone": {"type": "string"},
                "address": {"type": "string"},
                "doctor": {"type": "string"},
                "time": {"type": "string"},
            },
            required=["patient_name", "phone", "doctor", "time"],
        ),
        FunctionSchema(
            name="cancel_appointment",
            description=(
                getattr(cancel_appointment, "description", None)
                or "Cancel an appointment by ID or by patient phone."
            ),
            properties={
                "appointment_id": {"type": "string", "description": "APT-0001 when known."},
                "phone": {"type": "string", "description": "Patient phone if appointment id unknown."},
            },
            required=[],
        ),
        FunctionSchema(
            name="reschedule_appointment",
            description=(
                getattr(reschedule_appointment, "description", None)
                or "Reschedule an appointment to a new time (by id or phone)."
            ),
            properties={
                "appointment_id": {"type": "string", "description": "APT-0001 when known."},
                "new_time": {"type": "string", "description": "New day/time, e.g. tomorrow 4:50 PM."},
                "phone": {"type": "string", "description": "Patient phone if appointment id unknown."},
            },
            required=["new_time"],
        ),
        FunctionSchema(
            name="get_prescriptions",
            description=(
                getattr(get_prescriptions, "description", None)
                or "Look up prescribed medicines for a patient by patient_id, phone, or name."
            ),
            properties={
                "patient_id": {"type": "string", "description": "Patient id like PAT-0001 if known."},
                "phone": {"type": "string", "description": "Patient phone (required with name when id unknown)."},
                "patient_name": {"type": "string", "description": "Full name (required with phone when id unknown)."},
            },
            required=[],
        ),
    ]


def register_hospital_tools_on_llm(llm) -> list:
    """Register Tools.py handlers on a Pipecat LLM (used by Realtime)."""
    from pipecat.services.llm_service import FunctionCallParams

    schemas = hospital_function_schemas()

    for name, lc_tool in _LC_TOOLS.items():

        async def _handler(params: FunctionCallParams, _tool=lc_tool, _name=name):
            import time

            args = dict(params.arguments or {})
            logger.info(f"Realtime tool {_name} args={args}")
            t0 = time.perf_counter()
            error = ""
            data: Any = None
            try:
                raw = await asyncio.to_thread(_tool.invoke, args)
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    data = {"raw": raw}
                await params.result_callback(data)
            except Exception as e:
                logger.exception(f"Tool {_name} failed")
                error = str(e)
                data = {"ok": False, "message": error}
                await params.result_callback(data)
            try:
                from conversation_log import get_current_call_id, record_tool_call

                record_tool_call(
                    tool=_name,
                    arguments=args,
                    result=data if not error else None,
                    source="realtime",
                    call_id=get_current_call_id() or None,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    ok=False if error else None,
                    error=error,
                    tool_call_id=str(getattr(params, "tool_call_id", "") or ""),
                )
            except Exception:
                pass

        llm.register_function(name, _handler)

    return schemas


REALTIME_SYSTEM = (
    "You are a warm hospital front-desk assistant on a live voice call. "
    "Sound like a real person — soft, brief, never like a menu. "
    "FLOW: greet first ('Hi, how can I help you today?'); if need unclear ask "
    "'What brings you in today?' — NEVER list book/cancel/reschedule/prescriptions. "
    "When intent is clear from natural speech, acknowledge and ask phone; confirm phone; "
    "confirm name from lookup or collect+save for new patients; then help using intent + history. "
    "If they already have a visit, tell them and offer to keep/move/cancel. "
    "If not, offer to set one up softly. Medicine/timing questions → get_prescriptions. "
    "Use tools for lookup, booking, cancel, reschedule, doctors, prescriptions — never invent IDs. "
    "One short sentence per turn. Good: 'What brings you in today?' / "
    "'Sorry about that. What's your phone number?' "
    "Never identify a patient from chat memory alone. Never look up by name alone. "
    "After verified registration, call lookup_patient with department and follow booking_guidance. "
    "NEVER ask for DOB, age, or government ID. No markdown, bullets, or emojis."
)
