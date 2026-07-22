"""Unit tests for phone-fragment detection / merge helpers (STT text only)."""

from __future__ import annotations

from voice_bridge import (
    bot_awaiting_phone,
    looks_like_phone_fragment,
    merge_utterances,
    transcript_phone_digits,
)


def test_transcript_spoken_digits():
    assert transcript_phone_digits("nine eight five") == "985"
    assert transcript_phone_digits("It's nine eight five") == "985"
    assert transcript_phone_digits("zero four eight one nine seven") == "048197"
    assert transcript_phone_digits("nine eight five double zero") == "98500"
    assert transcript_phone_digits("(985), 004-8197.") == "9850048197"


def test_merge_utterances():
    assert merge_utterances("nine eight five", "zero four eight") == (
        "nine eight five zero four eight"
    )
    assert merge_utterances("hello", "hello") == "hello"
    assert merge_utterances("", "hi") == "hi"


def test_looks_like_phone_fragment_incomplete_only():
    assert looks_like_phone_fragment("nine eight five")
    assert looks_like_phone_fragment("It's nine eight five")
    assert looks_like_phone_fragment("double zero")
    assert looks_like_phone_fragment("0048197")
    # Complete 10-digit → send immediately (not a fragment to buffer)
    assert not looks_like_phone_fragment("nine eight five zero zero four eight one nine seven")
    assert not looks_like_phone_fragment("9850048197")
    # Normal speech
    assert not looks_like_phone_fragment("I want to book tomorrow")
    assert not looks_like_phone_fragment("Okay")
    assert not looks_like_phone_fragment("Why are you not listening to me?")


def test_bot_awaiting_phone():
    assert bot_awaiting_phone("Sure, what's your phone number?")
    assert bot_awaiting_phone("Could you please tell me the complete number?")
    assert not bot_awaiting_phone("Hi, how can I help you today?")
    assert not bot_awaiting_phone("Which doctor would you like?")
