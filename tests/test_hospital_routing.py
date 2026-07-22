"""Unit tests for hospital intent routing helpers."""

from __future__ import annotations

from hospital_routing import (
    _is_greeting_only,
    _is_short_affirmation,
    _sticky_route_from_history,
    _wants_prescriptions,
)


def test_greeting_only():
    assert _is_greeting_only("hi")
    assert _is_greeting_only("Hello!")
    assert _is_greeting_only("good morning")
    assert not _is_greeting_only("hi I need a doctor")


def test_short_affirmation():
    assert _is_short_affirmation("yes")
    assert _is_short_affirmation("Yeah.")
    assert _is_short_affirmation("sure please")
    assert not _is_short_affirmation("yes book tomorrow at 10")


def test_prescription_vs_department_medicine():
    assert _wants_prescriptions("what is my prescription dosage")
    assert _wants_prescriptions("how do I take my medicines")
    assert not _wants_prescriptions("I need family medicine")
    assert not _wants_prescriptions("book internal medicine")


def _msgs(*pairs):
    """Build simple (role, text) message list."""
    out = []
    for role, text in pairs:
        out.append((role, text))
    return out


def test_sticky_greeting_resets_to_general():
    messages = _msgs(
        ("user", "my stomach hurts"),
        ("assistant", "What's your phone number?"),
        ("user", "hello"),
    )
    assert _sticky_route_from_history(messages, last_agent="booking") == "general"


def test_sticky_keeps_booking_on_affirmation():
    messages = _msgs(
        ("user", "I need a dentist"),
        ("assistant", "Are you Eve?"),
        ("user", "yes"),
    )
    assert _sticky_route_from_history(messages, last_agent="booking") == "booking"


def test_sticky_symptoms_route_booking():
    messages = _msgs(("user", "I have a toothache"))
    assert _sticky_route_from_history(messages, last_agent="") == "booking"


def test_sticky_cancel_intent():
    messages = _msgs(("user", "I want to cancel my appointment"))
    assert _sticky_route_from_history(messages, last_agent="booking") == "cancelling"
