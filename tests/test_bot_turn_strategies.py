"""Ensure voice turn-stop honors stop_secs (not LocalSmartTurn)."""

from __future__ import annotations

from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from bot import _clamp_stop_secs, _user_turn_strategies, _vad_params
from soft_speech_turn_start import SoftSpeechUserTurnStartStrategy


def test_clamp_stop_secs():
    assert _clamp_stop_secs(None) == 0.2
    assert _clamp_stop_secs(0.2) == 0.2
    assert _clamp_stop_secs(0.01) == 0.1
    assert _clamp_stop_secs(99) == 3.0


def test_user_turn_strategies_use_speech_timeout_not_smart_turn():
    strategies = _user_turn_strategies(0.2)
    assert len(strategies.start) == 2
    assert isinstance(strategies.start[0], VADUserTurnStartStrategy)
    assert isinstance(strategies.start[1], SoftSpeechUserTurnStartStrategy)
    assert len(strategies.stop) == 1
    stop = strategies.stop[0]
    assert isinstance(stop, SpeechTimeoutUserTurnStopStrategy)
    assert not isinstance(stop, TurnAnalyzerUserTurnStopStrategy)
    assert stop._user_speech_timeout == 0.2


def test_user_turn_strategies_honor_slider():
    strategies = _user_turn_strategies(1.25)
    assert strategies.stop[0]._user_speech_timeout == 1.25
    vad = _vad_params(1.25)
    assert vad.stop_secs == 1.25


def test_soft_speech_blocked_when_bot_busy():
    state = {"busy": True, "bot_speaking": False}
    strategies = _user_turn_strategies(0.2, shared_state=state)
    soft = strategies.start[1]
    assert soft._is_blocked() is True
    state["busy"] = False
    assert soft._is_blocked() is False
    state["bot_speaking"] = True
    assert soft._is_blocked() is True
