"""End-of-turn method selection (turn_detection.py)."""

from __future__ import annotations

from pipecat.turns.user_start import ExternalUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

import turn_detection as td


def test_silence_uses_end_of_speech_wait():
    setup = td.build_turn_setup("silence", stop_secs=0.5)
    assert setup.detector == "silence"
    assert setup.vad.stop_secs == 0.5
    assert isinstance(setup.strategies.start[0], VADUserTurnStartStrategy)
    stop = setup.strategies.stop[0]
    assert isinstance(stop, SpeechTimeoutUserTurnStopStrategy)
    assert setup.speculation is None
    assert setup.uses_flux is False


def test_smart_turn_asks_model_at_short_pause():
    setup = td.build_turn_setup("smart_turn", stop_secs=1.0, max_silence_secs=1.5)
    assert setup.detector == "smart_turn"
    assert setup.vad.stop_secs == td.MODEL_VAD_STOP_SECS
    assert isinstance(setup.strategies.stop[0], TurnAnalyzerUserTurnStopStrategy)
    assert setup.max_silence_secs == 1.5


def test_flux_owns_turn_start_and_stop():
    setup = td.build_turn_setup("flux", stop_secs=0.2, speculative=True)
    assert setup.uses_flux is True
    assert isinstance(setup.strategies.start[0], ExternalUserTurnStartStrategy)
    assert isinstance(setup.strategies.stop[0], td.FluxTurnStopStrategy)
    assert setup.speculation == "external"


def test_speculation_signal_follows_detector():
    assert td.build_turn_setup("silence", stop_secs=0.2, speculative=True).speculation == "vad"
    assert td.build_turn_setup("smart_turn", stop_secs=0.2, speculative=True).speculation == "vad"


def test_unavailable_detector_falls_back_to_silence(monkeypatch):
    monkeypatch.setattr(
        td, "detector_unavailable_reason", lambda d: "missing" if d == "vap" else None
    )
    setup = td.build_turn_setup("vap", stop_secs=0.3)
    assert setup.detector == "silence"
    assert setup.vad.stop_secs == 0.3


def test_unknown_detector_name_is_silence():
    assert td.normalize_detector("nonsense") == "silence"
    assert td.normalize_detector(" Smart_Turn ") == "smart_turn"


def test_clamps():
    assert td.clamp_stop_secs(0.01) == 0.1
    assert td.clamp_stop_secs(9) == 3.0
    assert td.clamp_max_silence_secs(0) == 0.3
    assert td.clamp_max_silence_secs("x") == td.MAX_SILENCE_SECS_DEFAULT
