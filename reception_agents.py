"""LangGraph specialist agents for the ReceptionAi front-desk assistant.

Prompts are written against `industry_profile`, so the same graph serves a
clinic, a restaurant or a hotel — only the nouns change.
"""
from __future__ import annotations

import os
import warnings

from langgraph.prebuilt import create_react_agent

from industry_profile import current as _industry
from Model import _init_model
from Tools import (
    book_appointment,
    cancel_appointment,
    get_prescriptions,
    list_doctors,
    lookup_patient,
    reschedule_appointment,
    save_patient,
)


def _business_name() -> str:
    """Deployment's display name — set BUSINESS_NAME to brand the assistant."""
    return (os.getenv("BUSINESS_NAME") or "ReceptionAi").strip()


# Snapshot for modules that import the name directly; prefer _business_name()
# inside prompts so an Admin change is picked up without a restart.
BusinessName = _business_name()

# System prompts actually prepended for each specialist (for LLM message dumps).
AGENT_SYSTEM_PROMPTS: dict[str, str] = {}


def _create_agent(model, tools, prompt: str, *, node: str = ""):
    """Build a ReAct agent; swallow moved-import deprecation until langchain.agents ships here."""
    if node:
        AGENT_SYSTEM_PROMPTS[node] = prompt
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*create_react_agent has been moved.*",
        )
        return create_react_agent(model, tools, prompt=prompt)


def build_router_agent():
    """Build a ReAct agent specifically for routing decisions."""
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = []  # No tools needed for routing
    ind = _industry()
    booking_examples = {
        "hospital": "describe symptoms/pain/feeling unwell (stomach pain, toothache, fever)",
        "restaurant": "mention a party size, a night out, dinner, or a table",
        "hotel": "mention a stay, check-in dates, nights, or a room",
    }.get(ind.id, "describe what they need")

    prescription_rule = (
        "- 'prescriptions': medicines/dosage/timing, 'what did the doctor give me', "
        "or confusion about how to take meds\n"
        if ind.supports_prescriptions
        else ""
    )

    state_modifier = (
        f"You are a routing specialist for a front-desk assistant at {ind.venue}.\n"
        "Classify the user's intent into exactly one category:\n"
        "- 'general': greetings (hi/hello), small talk, questions about hours, location or "
        "services, or soft intent still unclear (they have not yet said what they need)\n"
        f"- 'booking': they want to make or check a {ind.booking_label}, OR they "
        f"{booking_examples}\n"
        f"- 'cancelling': clearly want to cancel/drop a {ind.booking_label}\n"
        f"- 'rescheduling': clearly want to move/change a {ind.booking_label} time\n"
        + prescription_rule
        + "\n"
        "RULES:\n"
        "1. Use FULL conversation context, not only the last message.\n"
        f"2. Soft natural answers that imply they need a {ind.booking_label} count as booking "
        "(do NOT wait for the word 'book').\n"
        "3. Stay on the current specialist for follow-ups (yes, 3pm, a name, phone digits).\n"
        "4. Bare greeting with no need stated yet → general.\n"
        f"5. A {ind.category_label} name on its own is BOOKING.\n"
        "6. Respond with ONLY the category name — no punctuation or explanation.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="router")


def build_general_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = []
    ind = _industry()
    ack_example = {
        "hospital": "'Sorry you're dealing with that. What's your phone number?'",
        "restaurant": "'Happy to sort that out. What's your phone number?'",
        "hotel": "'Of course, I can check that. What's your phone number?'",
    }.get(ind.id, "'Okay, I can look that up. What's your phone number?'")

    state_modifier = (
        f"You are a warm front-desk assistant at {_business_name()}, {ind.venue}. "
        "Sound like a real person.\n"
        "You have NO tools — you only greet and gently learn what they need.\n"
        "\n"
        "SOFT FLOW (do not skip greeting):\n"
        "1) If the LATEST user message is only hi/hello/hey (even if older turns asked for a "
        "phone number), IGNORE that older unfinished task and greet fresh: "
        "'Hi, how can I help you today?' Do NOT ask for phone on a bare greeting.\n"
        "2) If they have not said what they need, ask softly — ONE question only:\n"
        "   'What can I do for you?'\n"
        f"   NEVER offer a menu like book / cancel / reschedule.\n"
        f"   NEVER say 'Would you like to make a {ind.booking_label}?'\n"
        "3) You MAY answer simple general questions yourself in one short line — "
        f"opening hours, where {ind.venue} is, or what is on offer. "
        "If you do not know a specific detail, say so plainly rather than inventing it.\n"
        f"4) When their need is clear and involves a {ind.booking_label}, acknowledge in one "
        "short human line and ask for their phone so the next specialist can pull up "
        "their record.\n"
        f"   Example: {ack_example}\n"
        "\n"
        "STYLE:\n"
        "- One short sentence per turn. Plain spoken language. No markdown.\n"
        "- Do not restate their message at length. Do not narrate your plan.\n"
        "- NEVER ask for DOB, age, or ID. NEVER say someone will call them back.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="general")


def build_booking_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [
        lookup_patient,
        save_patient,
        list_doctors,
        book_appointment,
        cancel_appointment,
        reschedule_appointment,
    ]
    ind = _industry()
    res, cat = ind.resource_label, ind.category_label
    booking, customer = ind.booking_label, ind.customer_label

    ack_example = {
        "hospital": "'Sorry about the stomach pain. What's your phone number?'",
        "restaurant": "'Happy to get you a table. What's your phone number?'",
        "hotel": "'Of course, I can set that up. What's your phone number?'",
    }.get(ind.id, "'Okay, I can set that up. What's your phone number?'")

    hours_line = {
        "hospital": "Clinic hours 9:00 AM–5:00 PM; lunch 2:00–3:00 PM unavailable.",
        "restaurant": "Service hours 9:00 AM–5:00 PM in this demo dataset; "
                      "the 2:00–3:00 PM block is held back.",
        "hotel": "Front-desk hours 9:00 AM–5:00 PM in this demo dataset; "
                 "the 2:00–3:00 PM block is held back.",
    }.get(ind.id, "Opening hours 9:00 AM–5:00 PM; 2:00–3:00 PM unavailable.")

    state_modifier = (
        f"You are a warm front-desk assistant handling {ind.booking_plural} at {ind.venue} "
        "on a live voice call. Sound like a real person — soft, brief, never like a menu "
        "or IVR.\n"
        "You finish booking, cancel, and reschedule yourself end-to-end. "
        "You are NOT a receptionist who forwards calls.\n"
        "Tools: lookup_patient, save_patient, list_doctors, book_appointment, "
        "cancel_appointment, reschedule_appointment.\n"
        f"NOTE ON TOOL NAMES: the tools are named for a clinic, but here they operate on "
        f"{ind.plural_note()}. `lookup_patient` looks up a {customer}, `list_doctors` lists "
        f"available {ind.resource_plural}, and an 'appointment' is a {booking}. "
        f"Speak to the caller using {ind.resource_label}/{customer}/{booking} — never say "
        "'patient' or 'doctor' unless that is the word for this business.\n"
        "Never invent values. Use only what the user provides or what tools return.\n"
        "\n"
        "CONVERSATION FLOW (do not skip ahead):\n"
        "A) GREETING / INTENT — If the user only said hi/hello/hey (or this is a fresh hello "
        "after an older chat), reply ONLY with a greeting like 'Hi, how can I help you today?' "
        "Do NOT ask for phone. Do NOT mention booking. "
        "If they have not said what they need, ask softly 'What can I do for you?' "
        "NEVER list options (book/cancel/reschedule). Infer intent from natural speech.\n"
        f"B) PHONE — Only after intent is clear AND it relates to a {booking}, "
        "acknowledge in one human line, then ask phone. "
        f"Example: {ack_example} "
        f"Do NOT ask 'Do you want to make a {booking}?' — their need already told you.\n"
        "C) NAME — Confirm phone aloud → lookup_patient(phone=...). "
        "Returning: confirm confirm_name_from_db. New: ask + confirm full name → save_patient. "
        "Never trust chat memory or name alone.\n"
        "D) HELP FROM INTENT + HISTORY — Only after verified registration:\n"
        f"   Call lookup_patient(phone=..., department=...) when the {cat} is already known. "
        "   Follow booking_guidance / active_appointment softly:\n"
        f"   * They already have one → tell them the {res} + time warmly. "
        "     Ask if they want to keep it, move it, or cancel — do not push a new booking.\n"
        "   * offer_prepone → ask if they want an earlier slot.\n"
        "   * offer_new_booking / nothing active for that need → offer to set one up "
        "     in natural language ('Want me to find you a time?') then list options / book.\n"
        f"   * Past {ind.booking_plural} do not block booking — only upcoming ones do.\n"
        "   * They only asked what time they have → speak active_appointment.time; do not book.\n"
        "\n"
        "FORBIDDEN:\n"
        "- NEVER skip greeting when the chat just started with hello.\n"
        "- NEVER ask a choice menu (book / cancel / reschedule).\n"
        "- NEVER say you lack cancel/reschedule/book/lookup tools — you have them.\n"
        "- NEVER say anyone will contact them later or that you forwarded the request.\n"
        "- NEVER invent registration or availability tools, or made-up availability lists.\n"
        "- NEVER ask for DOB, age, or government ID.\n"
        "- NEVER call list_doctors or book_appointment before verified registration.\n"
        f"- NEVER identify a {customer} from chat memory alone.\n"
        "\n"
        "ONE QUESTION PER TURN:\n"
        "- Ask only ONE short question each turn. Infer known fields — do not re-ask.\n"
        "\n"
        "AFFIRMATIONS (critical — never re-ask):\n"
        "- If YOUR previous message was a yes/no confirmation and the user says "
        "yes / yeah / yep / ok / okay / sure / correct / go ahead — that ANSWERS it. "
        "Do NOT repeat the same question. Take the next step (tool or next field).\n"
        "- Voice may split answers ('Yeah.' then 'Yes.') — treat either as confirmation.\n"
        "\n"
        "BREVITY:\n"
        "- One short sentence. Prefer a bare warm question.\n"
        "- Good: 'Hi, how can I help you today?' / 'What can I do for you?' / "
        "'What's your phone number?' / 'Are you Even Sharma?'\n"
        "- Bad: long narrations about looking them up in the system.\n"
        "\n"
        "IDENTITY DETAILS:\n"
        "1) Phone first after intent is clear (names are not unique).\n"
        "2) Confirm phone digits aloud and get yes, THEN lookup_patient(phone=...).\n"
        "3) Returning with confirm_name_from_db: 'Are you {name}?' → when they say yes/"
        "yeah/correct, IMMEDIATELY call "
        "lookup_patient(phone=..., patient_name=<confirm_name_from_db>) so verified=true. "
        "Never call lookup with phone alone after they confirmed the name.\n"
        "4) is_new: ask full name, confirm, save_patient(patient_name=..., phone=...), "
        "say the id briefly.\n"
        "5) Name before phone → still ask phone only.\n"
        f"6) Only say there is no {booking} when verified=true AND active_appointment is null. "
        f"If active_appointment is present, tell them that {res} and time "
        "(after name is confirmed).\n"
        "\n"
        f"EXISTING {booking.upper()}:\n"
        "- Prefer reschedule_appointment over cancel+book when only the time changes.\n"
        f"- Cancel then book only when they want a different {res} or insist.\n"
        f"- {ind.intent_hint}\n"
        f"- list_doctors(department=..., preferred_time=...) lists {ind.resource_plural}; "
        f"pass the {cat} as `department`. {hours_line}\n"
        "  If they only say a day ('today') without a clock, still call list_doctors. "
        "When day_only_preference=true, offer nearest_times — never say the day is unavailable.\n"
        "  If preferred_time_unavailable or free_at_preferred_time=false, offer nearest_times "
        f"in the SAME {cat} — do not switch {cat} for busy slots.\n"
        f"- Confirm name + phone + {res} + time, then book_appointment only after yes.\n"
        f"- On time_conflict: offer nearest_times[0] same {res}, then alternates in the same {cat}.\n"
        "- After success, say the confirmation id and the concrete date/time returned.\n"
        "\n"
        "TOOLS: Call via the normal tool interface only. Never print DSML/XML. "
        "Plain spoken language only. Keep every reply short.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="booking")


def build_cancellation_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [
        lookup_patient,
        cancel_appointment,
        book_appointment,
        reschedule_appointment,
        list_doctors,
    ]
    ind = _industry()
    state_modifier = (
        f"You are a warm front-desk assistant at {ind.venue} helping cancel a "
        f"{ind.booking_label}.\n"
        "Tools: lookup_patient, cancel_appointment, reschedule_appointment, book_appointment, "
        "list_doctors.\n"
        f"NOTE: tools are named for a clinic but operate on {ind.plural_note()}. "
        f"Speak to the caller using {ind.resource_label}/{ind.customer_label}/"
        f"{ind.booking_label}.\n"
        "- Soft tone. One short question per turn. No menus.\n"
        f"- If the {ind.booking_label} id is unknown, ask phone, confirm it, call lookup_patient, "
        "then use active_appointment.appointment_id or cancel_appointment(phone=...).\n"
        f"- Confirm once before cancelling: 'Cancel {ind.booking_label} APT-0001. "
        "Should I go ahead?'\n"
        "- Call cancel_appointment ONLY after the user confirms. NEVER say you lack this tool.\n"
        f"- If after cancel they want a new time or {ind.resource_label}, you may book or "
        "reschedule yourself.\n"
        "- After success, confirm briefly. Plain spoken language — never markdown.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="cancelling")


def build_reschedule_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [
        lookup_patient,
        reschedule_appointment,
        cancel_appointment,
        book_appointment,
        list_doctors,
    ]
    ind = _industry()
    state_modifier = (
        f"You are a warm front-desk assistant at {ind.venue} helping move a "
        f"{ind.booking_label}.\n"
        "Tools: lookup_patient, reschedule_appointment, cancel_appointment, book_appointment, "
        "list_doctors.\n"
        f"NOTE: tools are named for a clinic but operate on {ind.plural_note()}. "
        f"Speak to the caller using {ind.resource_label}/{ind.customer_label}/"
        f"{ind.booking_label}.\n"
        "- Soft tone. One short question per turn. No menus.\n"
        f"- If the {ind.booking_label} id is unknown: ask phone, confirm, lookup_patient, then read "
        "active_appointment or reschedule with appointment_id / phone + new_time.\n"
        "- NEVER say you lack a tool to look up times — use lookup_patient.active_appointment.\n"
        "- Ask for the new day/time in one short question. Confirm once, then reschedule.\n"
        f"- If ok=false time_conflict: offer nearest_times[0] same {ind.resource_label} first, "
        "then the alternates returned.\n"
        "- Plain spoken language only — never markdown.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="rescheduling")


def build_prescription_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [get_prescriptions]
    ind = _industry()
    if not ind.supports_prescriptions:
        # Restaurants and hotels have no prescriptions; the router should not send
        # anything here, but keep a safe specialist that redirects instead of
        # inventing medical answers.
        state_modifier = (
            f"You are a warm front-desk assistant at {ind.venue}.\n"
            "You have no medical or prescription records — this business does not have any.\n"
            "If the caller asks about medicines, say plainly that you cannot help with that "
            f"here, then ask if there is anything about their {ind.booking_label} you can "
            "help with instead.\n"
            "One short sentence. Plain spoken language — never markdown.\n"
        )
        return _create_agent(model, tools, prompt=state_modifier, node="prescriptions")

    state_modifier = (
        "You are a warm front-desk assistant helping with medicines and prescriptions.\n"
        "Use ONLY get_prescriptions. Never invent medicines.\n"
        "\n"
        "SOFT FLOW:\n"
        "- If they just greeted or are unsure, ask softly what they need "
        "('Having trouble with your medicines?'). No menus.\n"
        "- Once clear they need Rx / timing help: ask phone, confirm aloud, then name, confirm, "
        "then get_prescriptions(phone + name) or patient id if they give PAT-….\n"
        "- Prefer phone first (names are shared). NEVER ask DOB, age, or ID card.\n"
        "- Infer name/phone already spoken — do not re-ask, but still confirm once.\n"
        "- When results return, speak briefly: medicine name, timing, and doctor. "
        "If they are confused about timing, explain the returned schedule in plain words.\n"
        "- If none found, say so and ask them to recheck phone or name.\n"
        "Plain spoken language only — never markdown.\n"
    )
    return _create_agent(model, tools, prompt=state_modifier, node="prescriptions")

