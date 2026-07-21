import os
import json
import re
import sys
from typing import Annotated, Any, List, TypedDict, Literal, NotRequired
import logging
from datetime import datetime
from pprint import pformat

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from langgraph.graph import StateGraph, END
from Model import _init_model

from Tools import (
    list_doctors,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
    get_prescriptions,
    lookup_patient,
    save_patient,
)
from agent_turn import run_turn

HospitalName = "DBC"
CHAT_LOG_PATH: str | None = None

logfile="app.log"
open(logfile,"w").close()
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def build_router_agent():
    """Build a ReAct agent specifically for routing decisions."""
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = []  # No tools needed for routing
    state_modifier = (
        "You are a routing specialist for a hospital front-desk assistant.\n"
        "Classify the user's intent into exactly one category:\n"
        "- 'general': greetings (hi/hello), small talk, or soft intent still unclear "
        "(they have not yet said what they need)\n"
        "- 'booking': they want to see a doctor / visit / check or manage a visit, OR they "
        "describe symptoms/pain/feeling unwell (stomach pain, toothache, headache, fever, etc.)\n"
        "- 'cancelling': clearly want to cancel/drop an appointment\n"
        "- 'rescheduling': clearly want to move/change an appointment time\n"
        "- 'prescriptions': medicines/dosage/timing, 'what did the doctor give me', "
        "or confusion about how to take meds\n"
        "\n"
        "RULES:\n"
        "1. Use FULL conversation context, not only the last message.\n"
        "2. Soft natural answers like 'my stomach hurts' or 'I need to see someone' → booking "
        "(do NOT wait for the word 'book').\n"
        "3. Stay on the current specialist for follow-ups (yes, 3pm, Dr. Smith, phone digits).\n"
        "4. Bare greeting with no need stated yet → general.\n"
        "5. Department names like 'Family Medicine' or 'Internal Medicine' are BOOKING, "
        "not prescriptions.\n"
        "6. Respond with ONLY the category name — no punctuation or explanation.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)


# -------- Router + Specialized Agents --------
def build_general_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = []
    state_modifier = (
        f"You are a warm front-desk assistant at {HospitalName}. Sound like a real person.\n"
        "You have NO tools — you only greet and gently learn what they need.\n"
        "\n"
        "SOFT FLOW (do not skip greeting):\n"
        "1) If the LATEST user message is only hi/hello/hey (even if older turns asked for a "
        "phone number), IGNORE that older unfinished task and greet fresh: "
        "'Hi, how can I help you today?' Do NOT ask for phone on a bare greeting.\n"
        "2) If they have not said what they need, ask softly — ONE question only:\n"
        "   'What brings you in today?' or 'What can I do for you?'\n"
        "   NEVER offer a menu like book / cancel / reschedule / prescriptions.\n"
        "   NEVER say 'Would you like to book an appointment?'\n"
        "3) When their need is clear from natural language (symptoms, see a doctor, "
        "existing visit, medicines, timing confusion), acknowledge in one short human line "
        "and ask for their phone so the next specialist can pull up their record.\n"
        "   Examples:\n"
        "   - 'Sorry you're dealing with that. What's your phone number?'\n"
        "   - 'Sure — I can help with your medicines. What's your phone number?'\n"
        "   - 'Okay, I can look that up. What's your phone number?'\n"
        "\n"
        "STYLE:\n"
        "- One short sentence per turn. Plain spoken language. No markdown.\n"
        "- Do not restate their message at length. Do not narrate your plan.\n"
        "- NEVER ask for DOB, age, or ID. NEVER say someone will call them back.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)



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
    state_modifier = (
        "You are a warm front-desk assistant for appointments on a live voice call. "
        "Sound like a real person — soft, brief, never like a menu or IVR.\n"
        "You finish booking, cancel, and reschedule yourself end-to-end. "
        "You are NOT a receptionist who forwards calls.\n"
        "Tools: lookup_patient, save_patient, list_doctors, book_appointment, "
        "cancel_appointment, reschedule_appointment.\n"
        "Never invent values. Use only what the user provides or what tools return.\n"
        "\n"
        "CONVERSATION FLOW (do not skip ahead):\n"
        "A) GREETING / INTENT — If the user only said hi/hello/hey (or this is a fresh hello "
        "after an older chat), reply ONLY with a greeting like 'Hi, how can I help you today?' "
        "Do NOT ask for phone. Do NOT mention booking. "
        "If they have not said what they need, ask softly 'What brings you in today?' "
        "NEVER list options (book/cancel/reschedule). Infer intent from natural speech.\n"
        "B) PHONE — Only after intent is clear AND it relates to a visit/symptoms/seeing a doctor, "
        "acknowledge in one human line, then ask phone. "
        "Example: 'Sorry about the stomach pain. What's your phone number?' "
        "Do NOT ask 'Do you want to book an appointment?' — their need already told you.\n"
        "C) NAME — Confirm phone aloud → lookup_patient(phone=...). "
        "Returning: confirm confirm_name_from_db. New: ask + confirm full name → save_patient. "
        "Never trust chat memory or name alone.\n"
        "D) HELP FROM INTENT + HISTORY — Only after verified registration:\n"
        "   Call lookup_patient(phone=..., department=...) when department is known from symptoms. "
        "   Follow booking_guidance / active_appointment softly:\n"
        "   * They already have a visit → tell them doctor + time warmly. "
        "     Ask if they want to keep it, move it, or cancel — do not push a new booking.\n"
        "   * offer_prepone → ask if they want an earlier slot.\n"
        "   * offer_new_booking / no active visit for that need → offer to set one up "
        "     in natural language ('Want me to find you a time?') then list doctors / book.\n"
        "   * Past visits do not block booking — only upcoming appointments do.\n"
        "   * They only asked what time they have → speak active_appointment.time; do not book.\n"
        "\n"
        "FORBIDDEN:\n"
        "- NEVER skip greeting when the chat just started with hello.\n"
        "- NEVER ask a choice menu (book / cancel / reschedule / prescriptions).\n"
        "- NEVER say you lack cancel/reschedule/book/lookup tools — you have them.\n"
        "- NEVER say anyone will contact them later or that you forwarded the request.\n"
        "- NEVER invent register_patient, get_available_slots, or made-up availability lists.\n"
        "- NEVER ask for DOB, age, or government ID.\n"
        "- NEVER call list_doctors or book_appointment before verified registration.\n"
        "- NEVER identify a patient from chat memory alone.\n"
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
        "- Good: 'Hi, how can I help you today?' / 'What brings you in today?' / "
        "'Sorry to hear that. What's your phone number?' / 'Are you Even Sharma?'\n"
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
        "say patient id briefly.\n"
        "5) Name before phone → still ask phone only.\n"
        "6) Only say there is no appointment when verified=true AND active_appointment is null. "
        "If active_appointment is present, tell them that doctor and time "
        "(after name is confirmed).\n"
        "\n"
        "EXISTING APPOINTMENT / BOOKING:\n"
        "- Prefer reschedule_appointment over cancel+book when only the time changes.\n"
        "- Cancel then book only when they want a different doctor or insist.\n"
        "- Department from symptoms (remember — do not re-ask if clear): "
        "tooth/dental → Dentistry; stomach/belly → Gastroenterology; "
        "headache/fever → General Medicine.\n"
        "- list_doctors(department=..., preferred_time=...). Clinic 9:00 AM–5:00 PM; "
        "lunch 2:00–3:00 PM unavailable.\n"
        "  If they only say a day ('today') without a clock, still call list_doctors. "
        "When day_only_preference=true, offer nearest_times — never say the day is unavailable.\n"
        "  If preferred_time_unavailable or free_at_preferred_time=false, offer nearest_times "
        "in the SAME department — do not switch departments for lunch/busy slots.\n"
        "- Confirm name + phone + doctor + time, then book_appointment only after yes.\n"
        "- On time_conflict: offer nearest_times[0] same doctor, then alternate_doctors same dept.\n"
        "- After success, say appointment id, patient id, and concrete date/time returned.\n"
        "\n"
        "TOOLS: Call via the normal tool interface only. Never print DSML/XML. "
        "Plain spoken language only. Keep every reply short.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)


def build_cancellation_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [
        lookup_patient,
        cancel_appointment,
        book_appointment,
        reschedule_appointment,
        list_doctors,
    ]
    state_modifier = (
        "You are a warm front-desk assistant helping cancel a visit.\n"
        "Tools: lookup_patient, cancel_appointment, reschedule_appointment, book_appointment, list_doctors.\n"
        "- Soft tone. One short question per turn. No menus.\n"
        "- If appointment id is unknown, ask phone, confirm it, call lookup_patient, then use "
        "active_appointment.appointment_id or cancel_appointment(phone=...).\n"
        "- Confirm once before cancelling: 'Cancel appointment APT-0001. Should I go ahead?'\n"
        "- Call cancel_appointment ONLY after the user confirms. NEVER say you lack this tool.\n"
        "- If after cancel they want a new time or doctor, you may book or reschedule yourself.\n"
        "- After success, confirm briefly. Plain spoken language — never markdown.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)


def build_reschedule_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [
        lookup_patient,
        reschedule_appointment,
        cancel_appointment,
        book_appointment,
        list_doctors,
    ]
    state_modifier = (
        "You are a warm front-desk assistant helping move a visit.\n"
        "Tools: lookup_patient, reschedule_appointment, cancel_appointment, book_appointment, list_doctors.\n"
        "- Soft tone. One short question per turn. No menus.\n"
        "- If appointment id is unknown: ask phone, confirm, lookup_patient, then read "
        "active_appointment or reschedule with appointment_id / phone + new_time.\n"
        "- NEVER say you lack a tool to look up times — use lookup_patient.active_appointment.\n"
        "- Ask for the new day/time in one short question. Confirm once, then reschedule.\n"
        "- If ok=false time_conflict: offer nearest_times[0] same doctor first, then alternate_doctors.\n"
        "- Plain spoken language only — never markdown.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)


def build_prescription_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [get_prescriptions]
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
    return create_react_agent(model, tools, prompt=state_modifier)


class GraphState(TypedDict):
    # add_messages keeps prior turns across invoke() calls (same thread_id).
    messages: Annotated[List[Any], add_messages]
    # Which specialist node produced the latest reply (for chat logs / admin).
    last_agent: NotRequired[str]


def _new_messages_only(prior: List[Any], result_messages: List[Any]) -> List[Any]:
    """Return only messages the sub-agent added (avoids duplicating history)."""
    if not result_messages:
        return []
    prior_len = len(prior or [])
    if len(result_messages) > prior_len:
        return list(result_messages[prior_len:])
    # Fallback: last message only if lengths match unexpectedly
    return list(result_messages[-1:])


def _message_plain(msg: Any) -> str:
    if isinstance(msg, tuple) and len(msg) >= 2:
        return str(msg[1] or "")
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts)
    return str(content or "")


def _message_role(msg: Any) -> str:
    if isinstance(msg, tuple) and msg:
        return str(msg[0] or "").lower()
    return str(getattr(msg, "type", getattr(msg, "role", "")) or "").lower()


def _is_short_affirmation(text: str) -> bool:
    t = (text or "").strip().lower().replace("!", "").replace("?", "")
    t = " ".join(t.replace(",", " ").split())
    if not t:
        return False
    # Normalize common multi-word confirms before token checks.
    for phrase in (
        "please go ahead",
        "go ahead please",
        "go ahead",
        "sounds good",
        "that's right",
        "thats right",
        "that is correct",
        "all right",
    ):
        t = t.replace(phrase, " yes ")
    t = " ".join(t.split())
    atoms = {
        "yes", "yeah", "yep", "yup", "ya", "yea", "ok", "okay", "sure",
        "correct", "right", "alright", "please",
    }
    if t in atoms:
        return True
    # "yeah. yes." / "yes yes" / "sure please"
    parts = [p for p in re.split(r"[.\s]+", t) if p]
    return bool(parts) and all(p in atoms for p in parts)


def _is_greeting_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        re.fullmatch(
            r"(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)"
            r"([!?.]|\s+(there|again))*[!?.\s]*",
            t,
        )
    )


_DEPT_MEDICINE_RE = re.compile(
    r"\b(?:family|internal|general|emergency|nuclear|preventive|sports)\s+medicine\b",
    re.IGNORECASE,
)
_PRESCRIPTION_INTENT_RE = re.compile(
    r"\b(?:prescriptions?|medications?|medicines?|dosage|how do i take|when do i take|what did the doctor)\b",
    re.IGNORECASE,
)


def _wants_prescriptions(text: str) -> bool:
    """True for medicine/Rx intent — not department names like 'family medicine'."""
    cleaned = _DEPT_MEDICINE_RE.sub(" ", text or "")
    return bool(_PRESCRIPTION_INTENT_RE.search(cleaned))


def _sticky_route_from_history(
    messages: List[Any],
    *,
    last_agent: str = "",
) -> Literal["general", "booking", "cancelling", "rescheduling", "prescriptions"] | None:
    """Keep mid-flow conversations on the specialist agent (esp. booking)."""
    recent = messages[-16:] if messages else []
    blob = " ".join(_message_plain(m).lower() for m in recent)
    last_user = ""
    for m in reversed(recent):
        if _message_role(m) in ("human", "user"):
            last_user = _message_plain(m).lower()
            break

    # Bare greetings ALWAYS reopen on general — even if an old thread still has
    # last_agent=booking / phone/pain markers in checkpoint history.
    if _is_greeting_only(last_user):
        return "general"

    if any(k in last_user for k in ("cancel appointment", "cancel my", "i want to cancel")):
        return "cancelling"
    if any(k in last_user for k in ("reschedule", "move my appointment", "change the time")):
        return "rescheduling"
    if _wants_prescriptions(last_user):
        return "prescriptions"

    prev = (last_agent or "").strip().lower()
    # Stay on the same specialist for yes/ok and other short mid-flow answers.
    # Do NOT treat greetings as mid-flow (handled above).
    if prev in ("booking", "cancelling", "rescheduling", "prescriptions"):
        if _is_short_affirmation(last_user):
            return prev  # type: ignore[return-value]
        words = last_user.split()
        if len(words) <= 6 and not any(
            k in last_user for k in ("cancel", "reschedule", "prescription")
        ) and not _wants_prescriptions(last_user):
            return prev  # type: ignore[return-value]

    booking_markers = (
        "book",
        "appointment",
        "phone",
        "number is",
        "full name",
        "just to confirm",
        "doctor",
        "department",
        "headache",
        "general medicine",
        "family medicine",
        "patient id",
        "tomorrow",
        " am",
        " pm",
        "pain",
        "stomach",
        "belly",
        "tooth",
        "dental",
        "fever",
        "sick",
        "hurt",
        "ache",
        "nausea",
        "vomit",
        "cough",
        "see a doctor",
        "need a doctor",
        "gastro",
    )
    # After soft intent is known (symptoms / visit talk), stay on booking —
    # unless still on general and they only answered a vague chit-chat line.
    if any(k in blob for k in booking_markers):
        if any(k in last_user for k in ("cancel", "reschedule")) or _wants_prescriptions(last_user):
            return None
        # Keep general for one soft "what brings you" answer only if no medical need yet.
        if prev == "general" and not any(k in last_user for k in _SYMPTOM_BOOKING_HINTS) and not any(
            k in last_user for k in ("doctor", "appointment", "visit", "see someone", "check up", "checkup")
        ):
            return "general"
        return "booking"
    return None


_SYMPTOM_BOOKING_HINTS = (
    "pain",
    "ache",
    "hurt",
    "fever",
    "sick",
    "stomach",
    "belly",
    "tooth",
    "dental",
    "headache",
    "nausea",
    "vomit",
    "cough",
    "see a doctor",
    "need a doctor",
    "unwell",
    "illness",
)


def build_graph():
    general_agent = build_general_agent()
    booking_agent = build_booking_agent()
    cancellation_agent = build_cancellation_agent()
    reschedule_agent = build_reschedule_agent()
    prescription_agent = build_prescription_agent()
    router_agent = build_router_agent()

    def route_decider(state: GraphState) -> Literal["general", "booking", "cancelling", "rescheduling", "prescriptions"]:
        logging.info(f"route_decider \n{pformat(state)}")
        all_messages = state.get("messages", [])

        sticky = _sticky_route_from_history(
            all_messages,
            last_agent=str(state.get("last_agent") or ""),
        )
        if sticky:
            logging.info(f"route_decider sticky: \n{sticky}")
            return sticky

        result = router_agent.invoke({"messages": all_messages}, config={"recursion_limit": 8})
        logging.info(f"route_decider result \n{pformat(result['messages'][-1])}")

        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        response_text = getattr(final_message, "content", "") if final_message else ""
        response_text = str(response_text).strip().lower()

        choice: str
        if _wants_prescriptions(response_text) or response_text.strip() in (
            "prescription",
            "prescriptions",
        ):
            choice = "prescriptions"
        elif "cancelling" in response_text or "cancel" in response_text:
            choice = "cancelling"
        elif "rescheduling" in response_text or "reschedule" in response_text:
            choice = "rescheduling"
        elif "booking" in response_text or "book" in response_text:
            choice = "booking"
        else:
            choice = "general"

        # Soft greeting stays on general; medical need → booking (no menu words required).
        last_user = ""
        for m in reversed(all_messages or []):
            if _message_role(m) in ("human", "user"):
                last_user = _message_plain(m).lower()
                break
        if _is_greeting_only(last_user):
            choice = "general"
        elif choice == "general" and any(
            k in last_user for k in ("book", "appointment", "doctor", "visit", "department", *_SYMPTOM_BOOKING_HINTS)
        ):
            choice = "booking"
        elif choice == "general" and _wants_prescriptions(last_user):
            choice = "prescriptions"
        # Never treat "Family Medicine" department requests as prescriptions.
        if choice == "prescriptions" and (
            "book" in last_user or "appointment" in last_user or "department" in last_user
        ) and not _wants_prescriptions(last_user):
            choice = "booking"

        # Prefer staying on last specialist for yes/ok if router drifted.
        prev = str(state.get("last_agent") or "").strip().lower()
        if prev in ("booking", "cancelling", "rescheduling", "prescriptions") and _is_short_affirmation(last_user):
            choice = prev

        logging.info(f"route_decider final: \n{choice}")
        return choice  # type: ignore[return-value]

    def router_node(state: GraphState) -> GraphState:
        logging.info("Running node: router")
        return state

    def _append_chat_record(node: str, state_messages: List[Any], result_messages: List[Any]) -> None:
        try:
            user_text = ""
            for msg in reversed(state_messages):
                role = getattr(msg, "type", getattr(msg, "role", None)) if not isinstance(msg, tuple) else msg[0]
                content = getattr(msg, "content", None) if not isinstance(msg, tuple) else msg[1]
                if role in ("human", "user"):
                    user_text = content if isinstance(content, str) else str(content)
                    break
            assistant_text = ""
            if result_messages:
                final = result_messages[-1]
                assistant_text = getattr(final, "content", None)
                if not isinstance(assistant_text, str):
                    assistant_text = str(final)
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "node": node,
                "user": user_text,
                "assistant": assistant_text,
            }
            if CHAT_LOG_PATH:
                with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except Exception as e:
            logging.error(f"Failed to append chat record: {e}")

    def general_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: general \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = general_agent.invoke({"messages": all_messages}, config={"recursion_limit": 12})
        logging.info(f"Running node: general result \n{pformat(result)}")
        msgs = result.get("messages", [])
        _append_chat_record("general", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "general"}

    def booking_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: booking \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = booking_agent.invoke({"messages": all_messages}, config={"recursion_limit": 20})
        msgs = result.get("messages", [])
        _append_chat_record("booking", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "booking"}

    def cancelling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: cancelling \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = cancellation_agent.invoke({"messages": all_messages}, config={"recursion_limit": 12})
        msgs = result.get("messages", [])
        _append_chat_record("cancelling", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "cancelling"}

    def rescheduling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: rescheduling \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = reschedule_agent.invoke({"messages": all_messages}, config={"recursion_limit": 16})
        msgs = result.get("messages", [])
        _append_chat_record("rescheduling", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "rescheduling"}

    def prescriptions_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: prescriptions \n{pformat(state)}")
        all_messages = state.get("messages", [])
        result = prescription_agent.invoke({"messages": all_messages}, config={"recursion_limit": 16})
        msgs = result.get("messages", [])
        _append_chat_record("prescriptions", all_messages, msgs)
        return {"messages": _new_messages_only(all_messages, msgs), "last_agent": "prescriptions"}

    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancelling", cancelling_node)
    graph.add_node("rescheduling", rescheduling_node)
    graph.add_node("prescriptions", prescriptions_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decider, {
        "general": "general",
        "booking": "booking",
        "cancelling": "cancelling",
        "rescheduling": "rescheduling",
        "prescriptions": "prescriptions",
    })
    # Let general immediately re-route in the same turn; specialists end the run
    # graph.add_edge("general", "router")
    graph.add_edge("general", END)
    graph.add_edge("booking", END)
    graph.add_edge("cancelling", END)
    graph.add_edge("rescheduling", END)
    graph.add_edge("prescriptions", END)
    
    # Add checkpointing with InMemorySaver
    return graph.compile(checkpointer=InMemorySaver())


def main():
    try:
        from service_settings import apply_settings_to_env

        apply_settings_to_env()
    except Exception:
        pass

    from conversation_log import _utc_iso, append_or_update_turn, end_call, start_call

    # Prepare per-session chat log file (legacy jsonl + structured call JSON)
    base_dir = os.path.dirname(__file__)
    chats_dir = os.path.join(base_dir, "chats")
    os.makedirs(chats_dir, exist_ok=True)
    session_name = datetime.now().strftime("chat-%Y%m%d-%H%M%S.jsonl")
    global CHAT_LOG_PATH
    CHAT_LOG_PATH = os.path.join(chats_dir, session_name)
    try:
        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except Exception as e:
        logging.error(f"Failed to initialize chat log file: {e}")

    app = build_graph()

    # Create a unique thread ID for this session
    thread_id = f"hospital_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    call_id = thread_id
    start_call(
        call_id,
        pipeline_mode="cli",
        session_id=thread_id,
        channel="cli",
        user_id="usr_cli",
        audio_codec="none",
    )

    def _logged_turn(user_input: str) -> str:
        user_sent = _utc_iso()
        append_or_update_turn(
            call_id,
            {
                "mode": "text",
                "input_type": "text",
                "user_text": user_input,
                "user_sent_at": user_sent,
                "bot_received_at": user_sent,
            },
            new_turn=True,
        )
        text_start = _utc_iso()
        turn = run_turn(app, user_input, thread_id, call_id=call_id)
        reply = turn.text if hasattr(turn, "text") else str(turn)
        agent_name = getattr(turn, "agent", "") or ""
        text_end = _utc_iso()
        append_or_update_turn(
            call_id,
            {
                "bot_text": reply,
                "agent_name": agent_name,
                "bot_text_first_token_at": text_start,
                "bot_text_first_shown_at": text_start,
                "bot_text_complete_at": text_end,
            },
            new_turn=False,
        )
        return reply

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        print(_logged_turn(user_input))
        end_call(call_id)
        return

    print("Hospital Agent ready. Type your request (Ctrl+C to exit).\n")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            print(_logged_turn(user_input))

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        end_call(call_id)


if __name__ == "__main__":
    main()