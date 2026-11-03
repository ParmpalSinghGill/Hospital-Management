"""End-of-turn detection for cascade voice calls: which method decides the caller
has finished speaking. Chosen in Admin → Settings → Voice turn-taking.

Four methods (ported from the ReceptionAi platform's ``packages/voice/turn.py``):

* ``silence``    — fixed silence after VAD stop (Admin "End-of-speech wait").
                   The original behaviour and the default.
* ``smart_turn`` — Pipecat Smart Turn v3: an audio model reads intonation and
                   rhythm at each pause and answers "finished" or "not yet".
* ``vap``        — Voice Activity Projection: predicts who speaks next; a
                   predicted shift to the agent ends the caller's turn. Needs
                   PyTorch and ``python setup_vap.py``.
* ``flux``       — Deepgram Flux: speech-to-text with its own end-of-turn model.

Silero VAD runs under every method. For the model methods its stop is a short
"ask the model now" pause, and the model, capped at ``max_silence_secs``,
decides.

Speculation (``voice_speculative``) starts the agent at the early signal — VAD's
pause, or Flux's EagerEndOfTurn — before the turn is confirmed. See
``speculation.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.user_start import ExternalUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from soft_speech_turn_start import SoftSpeechUserTurnStartStrategy

SILENCE = "silence"
SMART_TURN = "smart_turn"
VAP = "vap"
FLUX = "flux"
DETECTORS = (SILENCE, SMART_TURN, VAP, FLUX)

DETECTOR_LABELS = {
    SILENCE: "Silence timeout (End-of-speech wait)",
    SMART_TURN: "Smart Turn v3 (audio model)",
    VAP: "VAP (voice activity projection)",
    FLUX: "Deepgram Flux (STT end-of-turn)",
}

#: Silero stop for the model detectors: only marks *when to ask* the model.
#: 0.2s is the value Smart Turn was trained against.
MODEL_VAD_STOP_SECS = 0.2

#: Longest pause a model detector may call "not finished" before the turn is
#: ended anyway. Covers the ~0.9s gaps between spoken digit groups.
MAX_SILENCE_SECS_DEFAULT = 2.0

VAP_THRESHOLD = 0.5
FLUX_MODEL = "flux-general-en"
FLUX_EOT_THRESHOLD = 0.7
FLUX_EAGER_EOT_THRESHOLD = 0.5


def vap_model_dir() -> str:
    return os.getenv("VOICE_VAP_DIR", "~/.receptionai-models/vap")


def clamp_stop_secs(stop_secs: Any) -> float:
    """End-of-speech wait: 0.1–3.0s, default 0.2s (Pipecat stock VAD)."""
    try:
        stop = float(stop_secs) if stop_secs is not None else 0.2
    except (TypeError, ValueError):
        stop = 0.2
    return max(0.1, min(3.0, round(stop, 2)))


def clamp_max_silence_secs(value: Any) -> float:
    """Model detectors' "not finished yet" ceiling: 0.3–5.0s."""
    try:
        v = float(value) if value is not None else MAX_SILENCE_SECS_DEFAULT
    except (TypeError, ValueError):
        v = MAX_SILENCE_SECS_DEFAULT
    return max(0.3, min(5.0, round(v, 2)))


def normalize_detector(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in DETECTORS else SILENCE


def detector_unavailable_reason(detector: str) -> str | None:
    """Why ``detector`` cannot run on this host, or ``None`` when it can."""
    if detector == SMART_TURN:
        try:
            import onnxruntime  # noqa: F401
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (  # noqa: F401
                LocalSmartTurnAnalyzerV3,
            )
        except Exception as e:  # pragma: no cover - depends on host
            return f"Smart Turn needs onnxruntime: {e}"
    elif detector == FLUX:
        try:
            from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on host
            return f"Deepgram Flux is not in this Pipecat build: {e}"
    elif detector == VAP:
        from vap_turn import vap_unavailable_reason

        return vap_unavailable_reason(vap_model_dir())
    return None


def detector_options() -> list[dict[str, Any]]:
    """For the Admin dropdown: every method, with why it can't run (if so)."""
    out = []
    for d in DETECTORS:
        reason = detector_unavailable_reason(d)
        out.append(
            {"id": d, "label": DETECTOR_LABELS[d], "available": reason is None, "reason": reason or ""}
        )
    return out


def vad_params(stop_secs: float | None = None) -> VADParams:
    """Pipecat-default VAD start sensitivity; ``stop_secs`` from Admin settings."""
    return VADParams(
        confidence=0.7, start_secs=0.2, stop_secs=clamp_stop_secs(stop_secs), min_volume=0.6
    )


def silence_turn_strategies(
    stop_secs: float | None = None,
    *,
    shared_state: dict | None = None,
) -> UserTurnStrategies:
    """VAD + soft-speech STT start; speech-timeout stop (honors stop_secs).

    Pipecat default stop is LocalSmartTurnAnalyzerV3 (ignores ``stop_secs``).
    Recent sessions also showed STT finals in chat with **no** VAD turn start —
    only VAD start left those turns dead. SoftSpeech starts from finalized STT
    when the bot is idle; SpeechTimeout then ends after ``stop_secs``.
    """
    stop = clamp_stop_secs(stop_secs)
    state = shared_state if shared_state is not None else {}

    def _turn_start_blocked() -> bool:
        # Avoid cutting TTS / mid-agent audio with a late transcript turn.
        return bool(state.get("bot_speaking") or state.get("busy"))

    return UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(),
            SoftSpeechUserTurnStartStrategy(is_blocked=_turn_start_blocked),
        ],
        stop=[
            SpeechTimeoutUserTurnStopStrategy(
                user_speech_timeout=stop,
                wait_for_transcript=True,
            )
        ],
    )


class FluxTurnStopStrategy(ExternalUserTurnStopStrategy):
    """Pipecat's external stop strategy, without its 0.5s detour on Flux.

    Flux pushes the final transcript, then broadcasts UserStoppedSpeakingFrame.
    The stop frame is a system frame and overtakes the transcript, so the stock
    strategy sees "stopped" with no text and waits for its 0.5s settle timer.
    Flux only sends a final transcript at end of turn, so commit when it lands.
    """

    async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
        await super()._handle_transcription(frame)
        await self._maybe_trigger_user_turn_stopped()


@dataclass(frozen=True)
class TurnSetup:
    """Everything that differs between end-of-turn methods."""

    detector: str
    vad: VADParams
    strategies: UserTurnStrategies
    #: ``"vad"`` guesses at VAD's pause; ``"external"`` on Flux's early signal;
    #: ``None`` when speculation is off.
    speculation: str | None
    uses_flux: bool
    max_silence_secs: float


def build_turn_setup(
    detector: str,
    *,
    stop_secs: float | None,
    max_silence_secs: float | None = None,
    speculative: bool = False,
    shared_state: dict | None = None,
) -> TurnSetup:
    """Map a method to its VAD window, turn strategies and speculation signal.

    Falls back to ``silence`` (with a warning) when the chosen method cannot
    run here, so a missing model never fails a call at connect time.
    """
    detector = normalize_detector(detector)
    reason = detector_unavailable_reason(detector)
    if reason:
        logger.warning(f"Turn detector {detector!r} unavailable ({reason}); using silence")
        detector = SILENCE
    max_silence = clamp_max_silence_secs(max_silence_secs)

    if detector == FLUX:
        return TurnSetup(
            detector=FLUX,
            vad=vad_params(MODEL_VAD_STOP_SECS),
            strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[FluxTurnStopStrategy()],
            ),
            speculation="external" if speculative else None,
            uses_flux=True,
            max_silence_secs=max_silence,
        )

    base = silence_turn_strategies(stop_secs, shared_state=shared_state)
    if detector == SILENCE:
        return TurnSetup(
            detector=SILENCE,
            vad=vad_params(stop_secs),
            strategies=base,
            speculation="vad" if speculative else None,
            uses_flux=False,
            max_silence_secs=max_silence,
        )

    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

    params = SmartTurnParams(stop_secs=max_silence)
    if detector == VAP:
        from vap_turn import VapTurnAnalyzer

        analyzer: Any = VapTurnAnalyzer(
            model_dir=vap_model_dir(), threshold=VAP_THRESHOLD, params=params
        )
    else:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        analyzer = LocalSmartTurnAnalyzerV3(params=params)

    return TurnSetup(
        detector=detector,
        vad=vad_params(MODEL_VAD_STOP_SECS),
        strategies=UserTurnStrategies(
            start=list(base.start or []),
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)],
        ),
        speculation="vad" if speculative else None,
        uses_flux=False,
        max_silence_secs=max_silence,
    )


def build_flux_stt(*, keyterms: list[str], speculative: bool, max_silence_secs: float):
    """Deepgram Flux STT. EagerEndOfTurn is only requested when speculating."""
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

    return DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramFluxSTTService.Settings(
            model=os.getenv("DEEPGRAM_FLUX_MODEL", FLUX_MODEL),
            eot_threshold=FLUX_EOT_THRESHOLD,
            eager_eot_threshold=FLUX_EAGER_EOT_THRESHOLD if speculative else None,
            eot_timeout_ms=int(max_silence_secs * 1000),
            keyterm=list(keyterms),
        ),
    )
