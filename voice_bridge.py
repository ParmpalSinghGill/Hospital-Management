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
        LLMTextFrame,
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
    class GuardedLLMTextFrame(LLMTextFrame, UninterruptibleFrame):
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
            text = GuardedLLMTextFrame(reply)
            text.skip_tts = True
            end = GuardedLLMFullResponseEndFrame()
            end.skip_tts = True
            await self.push_frame(start, direction)
            await self.push_frame(text, direction)
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
            # Shield inference from Pipecat barge-in cancellation. A quick
            # follow-up utterance (e.g. duplicate "Yes.") otherwise cancels
            # this coroutine, drops the reply, and leaves the call silent —
            # even though the agent thread finished (save_patient, etc.).
            turn_task = asyncio.create_task(
                asyncio.to_thread(
                    run_turn,
                    graph_app,
                    user_text,
                    thread_id,
                    call_id=log_call_id,
                )
            )
            while not turn_task.done():
                try:
                    await asyncio.shield(turn_task)
                except asyncio.CancelledError:
                    logger.debug(
                        "Interrupt during LangGraph turn; keeping inference"
                    )
                    try:
                        from conversation_log import record_interrupt

                        record_interrupt(
                            log_call_id,
                            reason="interrupt_during_inference",
                            user_text=user_text,
                            bot_text=str(state.get("last_bot_text") or ""),
                            phase="agent_busy",
                            strategy="CancelledError",
                        )
                    except Exception:
                        pass
                    continue
            turn_result = turn_task.result()
            reply = getattr(turn_result, "text", None) or str(turn_result or "")
            agent = getattr(turn_result, "agent", "") or ""
            reply = sanitize_assistant_reply(reply or "")
            if not reply:
                reply = "Sorry, I couldn't process that. Could you say it again?"
            if agent:
                logger.info(f"LangGraph reply via [{agent}]: {reply!r}")
            else:
                logger.info(f"LangGraph reply: {reply!r}")
            try:
                await self._push_spoken_reply(reply, direction)
            except asyncio.CancelledError:
                # Still force the spoken reply out after a barge-in cancel.
                logger.debug("TTS push interrupted; retrying spoken reply once")
                try:
                    from conversation_log import record_interrupt

                    record_interrupt(
                        log_call_id,
                        reason="interrupt_during_tts_push",
                        user_text=user_text,
                        bot_text=reply,
                        phase="tts_push",
                        strategy="CancelledError",
                    )
                except Exception:
                    pass
                await self._push_spoken_reply(reply, direction)

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

                if self._busy:
                    self._pending_user_text = user_text
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
                    return

                self._busy = True
                state["busy"] = True
                try:
                    next_text = user_text
                    while next_text:
                        try:
                            await self._run_one_turn(next_text, direction)
                        except Exception as e:
                            logger.exception("LangGraph turn failed")
                            err = f"Sorry, something went wrong: {e}"
                            try:
                                await self._push_spoken_reply(err, direction)
                            except asyncio.CancelledError:
                                pass
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
            args = dict(params.arguments or {})
            logger.info(f"Realtime tool {_name} args={args}")
            try:
                raw = await asyncio.to_thread(_tool.invoke, args)
            except Exception as e:
                logger.exception(f"Tool {_name} failed")
                await params.result_callback({"ok": False, "message": str(e)})
                return
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                data = {"raw": raw}
            await params.result_callback(data)

        llm.register_function(name, _handler)

    return schemas


REALTIME_SYSTEM = (
    "You are a hospital appointment assistant on a live voice call. "
    "Sound natural and concise. Use tools for looking up patients, booking, cancelling, "
    "rescheduling, listing doctors, and prescriptions — never invent IDs, doctor names, or medicines. "
    "Ask for one missing field at a time. Always confirm phone and name out loud and wait for yes "
    "before lookup or booking. Patients often forget their id — verify with phone and name only. "
    "NEVER ask for date of birth, age, or government ID. If they offer DOB, say you only need phone and name. "
    "For returning patients, mention their last doctor and ask if they want the same one. "
    "If same doctor, only ask day and time. If a new health issue, pick another department. "
    "If they want a second opinion or a new doctor, list other doctors in the same department. "
    "Confirm results briefly. No markdown, bullets, or emojis."
)
