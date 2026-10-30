"""Industry vocabulary for ReceptionAi.

The booking engine underneath is industry-agnostic: a caller reserves time with a
named resource. Only the *words* change between a clinic, a restaurant, and a
hotel. This module holds those words in one place so agent prompts, greetings and
UI copy stay consistent no matter which vertical the deployment serves.

Selected with the `INDUSTRY` env var (`hospital` | `restaurant` | `hotel`), which
the Admin console writes through `service_settings.apply_settings_to_env`.

The database schema keeps its original column names (doctors, patients,
prescriptions) — renaming storage would break existing datasets and MCP clients
for no user-visible gain. `resource_label` / `customer_label` are how those rows
are *spoken about*.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_INDUSTRY = "hospital"


@dataclass(frozen=True)
class IndustryProfile:
    """Vocabulary and behaviour hints for one vertical."""

    id: str
    # Display
    label: str            # "Hospital / Clinic"
    tagline: str          # front page card copy
    venue: str            # "the clinic" — used in spoken lines
    # Nouns the agents speak
    resource_label: str        # doctor / table / room
    resource_plural: str       # doctors / tables / rooms
    customer_label: str        # patient / guest / guest
    booking_label: str         # appointment / reservation / stay
    booking_plural: str        # appointments / reservations / stays
    category_label: str        # department / seating area / room type
    # Extra guidance appended to specialist prompts
    intent_hint: str = ""
    # Departments/categories seeded or offered when the caller is vague
    categories: tuple[str, ...] = field(default_factory=tuple)
    # Whether the prescriptions specialist makes sense for this vertical
    supports_prescriptions: bool = False
    # Example prompts shown on the front page
    examples: tuple[str, ...] = field(default_factory=tuple)

    def plural_note(self) -> str:
        """Short phrase telling the LLM what the clinic-named tools act on here."""
        return f"{self.resource_plural} and {self.booking_plural}"


HOSPITAL = IndustryProfile(
    id="hospital",
    label="Hospital / Clinic",
    tagline="Appointments, cancellations and prescription questions for patients.",
    venue="the clinic",
    resource_label="doctor",
    resource_plural="doctors",
    customer_label="patient",
    booking_label="appointment",
    booking_plural="appointments",
    category_label="department",
    intent_hint=(
        "Callers often describe symptoms instead of naming a department. "
        "Map symptoms to a department yourself: tooth/dental -> Dentistry; "
        "stomach/belly -> Gastroenterology; headache/fever -> General Medicine."
    ),
    categories=(
        "General Medicine",
        "Dentistry",
        "Cardiology",
        "Orthopedics",
        "Gastroenterology",
    ),
    supports_prescriptions=True,
    examples=(
        "Book me with Cardiology tomorrow at 10am. My name is Priya.",
        "Which doctors are in Orthopedics?",
        "Cancel appointment APT-0012",
        "Reschedule APT-0012 to Friday at 3pm",
    ),
)

RESTAURANT = IndustryProfile(
    id="restaurant",
    label="Restaurant",
    tagline="Table reservations, changes and questions about hours or the menu.",
    venue="the restaurant",
    resource_label="table",
    resource_plural="tables",
    customer_label="guest",
    booking_label="reservation",
    booking_plural="reservations",
    category_label="seating area",
    intent_hint=(
        "Ask for party size early — it decides which tables fit. "
        "Callers may ask about seating preference (indoor, patio, bar, private). "
        "Treat a named seating area the same way a clinic treats a department."
    ),
    categories=("Main Dining", "Patio", "Bar", "Private Room"),
    supports_prescriptions=False,
    examples=(
        "Table for four on Friday at 7pm, name is Priya.",
        "Do you have anything on the patio tonight?",
        "Cancel reservation APT-0012",
        "Move my Friday booking to 8pm",
    ),
)

HOTEL = IndustryProfile(
    id="hotel",
    label="Hotel",
    tagline="Room bookings, date changes and front-desk questions for guests.",
    venue="the hotel",
    resource_label="room",
    resource_plural="rooms",
    customer_label="guest",
    booking_label="booking",
    booking_plural="bookings",
    category_label="room type",
    intent_hint=(
        "Ask for the check-in day and number of guests early. "
        "Callers may name a room type (deluxe, suite, twin). "
        "Treat a room type the same way a clinic treats a department."
    ),
    categories=("Standard", "Deluxe", "Suite", "Twin"),
    supports_prescriptions=False,
    examples=(
        "I need a deluxe room from Friday, name is Priya.",
        "What rooms are free this weekend?",
        "Cancel booking APT-0012",
        "Can I move my check-in to Saturday?",
    ),
)

PROFILES: dict[str, IndustryProfile] = {p.id: p for p in (HOSPITAL, RESTAURANT, HOTEL)}
INDUSTRY_IDS: tuple[str, ...] = tuple(PROFILES)


def get_profile(industry: str | None = None) -> IndustryProfile:
    """Resolve a profile by id, falling back to `INDUSTRY` env then hospital."""
    wanted = (industry or os.getenv("INDUSTRY") or DEFAULT_INDUSTRY).strip().lower()
    return PROFILES.get(wanted, PROFILES[DEFAULT_INDUSTRY])


def current() -> IndustryProfile:
    """Profile for this process, re-read each call so Admin changes take effect."""
    return get_profile()


def vocabulary(industry: str | None = None) -> dict[str, str]:
    """Flat noun map — handy for `str.format` in prompts and for the JSON APIs."""
    p = get_profile(industry)
    return {
        "industry": p.id,
        "label": p.label,
        "venue": p.venue,
        "resource": p.resource_label,
        "resources": p.resource_plural,
        "customer": p.customer_label,
        "booking": p.booking_label,
        "bookings": p.booking_plural,
        "category": p.category_label,
    }


def describe(industry: str | None = None) -> dict:
    """Full profile payload for the front page and admin UI."""
    p = get_profile(industry)
    return {
        "id": p.id,
        "label": p.label,
        "tagline": p.tagline,
        "venue": p.venue,
        "resource_label": p.resource_label,
        "resource_plural": p.resource_plural,
        "customer_label": p.customer_label,
        "booking_label": p.booking_label,
        "booking_plural": p.booking_plural,
        "category_label": p.category_label,
        "categories": list(p.categories),
        "supports_prescriptions": p.supports_prescriptions,
        "examples": list(p.examples),
    }


def all_profiles() -> list[dict]:
    return [describe(pid) for pid in INDUSTRY_IDS]
