"""Mute strategies that cannot leave the mic stuck off after a cut TTS reply."""

from __future__ import annotations

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
)
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


class SafeBotSpeakingMuteStrategy(BaseUserMuteStrategy):
    """Mute while the bot speaks, and always unmute if speech is interrupted.

    Pipecat's AlwaysUserMuteStrategy only clears on BotStoppedSpeakingFrame.
    If TTS is interrupted before that frame, the user stays muted forever —
    which feels like a sudden pin-drop silence for the rest of the call.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame) -> bool:
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, (BotStoppedSpeakingFrame, InterruptionFrame)):
            self._bot_speaking = False

        return self._bot_speaking
