"""Pipeline processor that records interrupt / speak timeline events with text."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class InterruptTraceProcessor(FrameProcessor):
    """Log timed interrupt/speak events with the latest user/bot text context."""

    def __init__(self, call_id: str, state: dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._call_id = call_id
        self._state = state

    def _snapshot(self) -> tuple[str, str, str]:
        user_text = str(self._state.get("last_user_text") or "")
        bot_text = str(self._state.get("last_bot_text") or "")
        if self._state.get("busy"):
            phase = "agent_busy"
        elif self._state.get("bot_speaking"):
            phase = "bot_speaking"
        else:
            phase = "idle"
        return user_text, bot_text, phase

    def _log(
        self,
        event_type: str,
        *,
        strategy: str = "",
        user_text: str = "",
        bot_text: str = "",
        phase: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        try:
            from conversation_log import record_timeline_event

            u, b, default_phase = self._snapshot()
            record_timeline_event(
                self._call_id,
                event_type=event_type,
                user_text=user_text or u,
                bot_text=bot_text or b,
                strategy=strategy,
                phase=phase or default_phase,
                source="server",
                extra=extra,
            )
        except Exception as e:
            logger.debug(f"timeline log failed: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._log(
                "interrupt",
                strategy="InterruptionFrame",
                extra={"direction": str(direction)},
            )
            logger.info(
                f"Interrupt traced ({self._call_id}): "
                f"phase={self._snapshot()[2]} "
                f"user={self._state.get('last_user_text')!r} "
                f"bot={(self._state.get('last_bot_text') or '')[:80]!r}"
            )
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._log("user_started_speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._log("user_stopped_speaking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._state["bot_speaking"] = True
            self._log("bot_started_speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._state["bot_speaking"] = False
            self._log("bot_stopped_speaking")
        elif isinstance(frame, TranscriptionFrame):
            text = (getattr(frame, "text", None) or "").strip()
            if text:
                # Keep freshest heard transcript for the next interrupt snapshot.
                self._state["last_heard_transcript"] = text
                # Only persist transcripts that arrive while the bot is active —
                # those are the usual interrupt triggers.
                if getattr(frame, "finalized", False) and (
                    self._state.get("bot_speaking") or self._state.get("busy")
                ):
                    self._log(
                        "transcript_during_bot",
                        user_text=text,
                        extra={"finalized": True},
                    )

        await self.push_frame(frame, direction)
