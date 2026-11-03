"""Speculative voice turns: start the agent before the caller's turn is confirmed.

When the caller pauses, VAD fires well before the end-of-turn detector commits
the turn. With speculation on, the voice bridge starts the LangGraph run at
that pause on the words heard so far. If the committed words match, the reply
is already (partly) computed; if the caller keeps talking or STT revises the
words, the guess is thrown away.

Two rules keep a wrong guess harmless:

* **It runs on a copy of the conversation.** The guess uses its own LangGraph
  thread seeded with the real thread's history, so a discarded guess leaves
  nothing behind. On a hit the copy becomes the conversation (see
  ``voice_bridge``).
* **Writes wait for confirmation.** ``book_appointment``, ``cancel_appointment``,
  ``reschedule_appointment`` and ``save_patient`` call :func:`hold_side_effects`
  first. During a guess that blocks until the turn is confirmed (then the write
  goes ahead) or abandoned (then it raises :class:`SpeculationAbandoned`, which
  is a ``BaseException`` so LangGraph's tool error handling cannot turn it into
  a message the model answers).
"""

from __future__ import annotations

import contextvars
import re
import threading
from collections.abc import Awaitable, Callable

__all__ = [
    "SpeculationAbandoned",
    "SpeculationGate",
    "UserTurnText",
    "hold_side_effects",
    "make_transcript_tap",
    "normalize_utterance",
    "speculation_gate",
]

#: How long a held write waits for the turn to be confirmed before giving up.
#: Longer than any end-of-turn ceiling (max 5s) plus STT latency.
_HOLD_TIMEOUT_S = 30.0

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")


class SpeculationAbandoned(BaseException):
    """A held write belonged to a guess that was thrown away."""


class SpeculationGate:
    """Released once a speculative turn is confirmed or abandoned.

    A ``threading.Event`` rather than an asyncio one: LangGraph runs the turn in
    a worker thread (``asyncio.to_thread``), and its tools in further threads.
    """

    def __init__(self, call_id: str = "") -> None:
        self._event = threading.Event()
        self.abandoned = False
        #: Restored on confirmation so a released write tags the live call log.
        self.call_id = call_id
        #: LLM calls this guess made (fed by llm_timing) — the cost of a miss.
        self.llm_calls = 0
        self.llm_ms = 0.0
        self._count_lock = threading.Lock()

    def note_llm_call(self, call) -> None:  # noqa: ANN001 - llm_timing.LLMCallTiming
        if getattr(call, "status", "") != "ok":
            return
        with self._count_lock:
            self.llm_calls += 1
            self.llm_ms += float(getattr(call, "total_ms", None) or 0.0)

    def confirm(self) -> None:
        self._event.set()

    def abandon(self) -> None:
        self.abandoned = True
        self._event.set()

    def wait(self, timeout: float = _HOLD_TIMEOUT_S) -> bool:
        return self._event.wait(timeout)


#: Set for the duration of a speculative run. Context variables are copied into
#: ``asyncio.to_thread`` workers and into LangGraph's tool executor threads.
speculation_gate: contextvars.ContextVar[SpeculationGate | None] = contextvars.ContextVar(
    "speculation_gate", default=None
)


def hold_side_effects() -> None:
    """Call first in any tool that writes. No-op outside a speculative run."""
    gate = speculation_gate.get()
    if gate is None:
        return
    if not gate.wait() or gate.abandoned:
        raise SpeculationAbandoned()
    if gate.call_id:
        try:
            from conversation_log import set_current_call_id

            set_current_call_id(gate.call_id)
        except Exception:
            pass


def normalize_utterance(text: str) -> str:
    """Compare utterances ignoring case, punctuation and spacing.

    Interim and final transcripts of the same words differ in exactly these
    ways ("Thursday at ten" / "Thursday at ten."), and a guess must not be
    thrown away over a full stop.
    """
    return _SPACE.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


class UserTurnText:
    """Finalized transcript segments plus the live interim, for the open turn."""

    def __init__(self) -> None:
        self._finals: list[str] = []
        self._interim = ""

    @property
    def text(self) -> str:
        parts = [*self._finals]
        if self._interim:
            parts.append(self._interim)
        return " ".join(p for p in parts if p).strip()

    def add_final(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._finals.append(text)
        self._interim = ""

    def set_interim(self, text: str) -> None:
        self._interim = (text or "").strip()

    def reset(self) -> None:
        self._finals.clear()
        self._interim = ""


def make_transcript_tap(
    turn_text: UserTurnText,
    *,
    on_update: Callable[[], Awaitable[None]] | None = None,
):
    """Pass-through processor between STT and the user aggregator.

    The aggregator consumes transcription frames, so the agent never sees the
    words of a turn until it is committed. Speculation needs them at the pause;
    this tap keeps a running copy. Imported lazily so this module stays
    importable without Pipecat (tools import :func:`hold_side_effects`).
    """
    from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class TranscriptTap(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
            await super().process_frame(frame, direction)
            changed = False
            if isinstance(frame, TranscriptionFrame):
                turn_text.add_final(frame.text)
                changed = True
            elif isinstance(frame, InterimTranscriptionFrame):
                turn_text.set_interim(frame.text)
                changed = True
            await self.push_frame(frame, direction)
            if changed and on_update is not None:
                await on_update()

    return TranscriptTap()

