"""Turn-start fallback when Silero VAD misses soft speech but STT still hears it."""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class SoftSpeechUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Start a user turn from a finalized STT transcript when VAD did not.

    Recent calls showed chat ``Hello?`` (RTVI user transcript) with **zero**
    ``user_started_speaking`` / ``user_turn_started`` events — Silero never
    opened a turn, so SpeechTimeout never ran and LangGraph never fired.

    Stock ``TranscriptionUserTurnStartStrategy`` also starts on *interim*
    transcripts and while the bot speaks (cutting TTS). This variant only
    uses finalized text and skips while the bot is busy/speaking.
    """

    def __init__(
        self,
        *,
        is_blocked: Callable[[], bool] | None = None,
        **kwargs,
    ):
        # Interruptions OK when idle; blocked path never triggers while bot speaks.
        super().__init__(enable_interruptions=True, enable_user_speaking_frames=True, **kwargs)
        self._is_blocked = is_blocked or (lambda: False)

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if not isinstance(frame, TranscriptionFrame):
            return ProcessFrameResult.CONTINUE

        text = (getattr(frame, "text", None) or "").strip()
        if not text:
            return ProcessFrameResult.CONTINUE

        # Deepgram pushes TranscriptionFrame only for is_final results.
        # Do NOT require frame.finalized=True — that flag is set only after a
        # VAD-driven Finalize request. Soft speech with no VAD would never start.
        if self._is_blocked():
            return ProcessFrameResult.CONTINUE

        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP
