"""Regression tests: browser latency is kept; demo sessions are not counted."""

from __future__ import annotations

import log_routes
from chat_log_query import _ms_between, _timing_samples_from_interactions


def test_event_body_keeps_latency_fields():
    body = log_routes.EventBody(
        call_id="sess_1234", type="bot_voice_first_heard", latency_ms=1870, first_speech="Hi"
    )
    dumped = body.model_dump()
    assert dumped["latency_ms"] == 1870
    assert dumped["first_speech"] == "Hi"


def test_latency_estimated_from_timestamps_when_missing():
    ix = {
        "client_timings": {"first_text": "Hello"},
        "timestamps": {
            "user_sent_time": "2026-10-30T15:50:20.000Z",
            "first_token_visible_to_user": "2026-10-30T15:50:21.500Z",
            "first_audio_heard_by_user": "2026-10-30T15:50:22.000Z",
        },
    }
    text, audio = _timing_samples_from_interactions([ix])
    assert text == [1500.0]
    assert audio == [2000.0]


def test_recorded_latency_wins_over_timestamps():
    ix = {
        "client_timings": {"first_text_latency_ms": 900},
        "timestamps": {
            "user_sent_time": "2026-10-30T15:50:20.000Z",
            "first_token_visible_to_user": "2026-10-30T15:50:25.000Z",
        },
    }
    text, _ = _timing_samples_from_interactions([ix])
    assert text == [900.0]


def test_implausible_gaps_ignored():
    assert _ms_between("2026-10-30T10:00:00Z", "2026-10-30T09:59:59Z") is None
    assert _ms_between("2026-10-30T10:00:00Z", "2026-10-30T10:05:00Z") is None
    assert _ms_between(None, "2026-10-30T10:00:00Z") is None
