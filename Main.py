import os
import json
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
        "You are a routing specialist for a hospital system.\n"
        "Your job is to classify user intent into exactly one of these categories:\n"
        "- 'booking': when user wants to book/schedule a new appointment\n"
        "- 'cancelling': when user wants to cancel/delete an appointment\n"
        "- 'rescheduling': when user wants to change/move an existing appointment's time\n"
        "- 'prescriptions': when user asks what medicine/drugs a doctor prescribed, dosage timing, or prescription details\n"
        "- 'general': for greetings, questions, or unclear intent\n"
        "\n"
        "IMPORTANT RULES:\n"
        "1. Consider the FULL conversation context, not just the last message\n"
        "2. If the conversation is already about booking/cancelling/rescheduling/prescriptions, stay in that context\n"
        "3. Follow-up questions like 'yes', '3pm', 'Dr. Smith' should maintain the current intent\n"
        "4. Only switch to 'general' if the user starts a completely new topic\n"
        "5. Respond with ONLY the category name (e.g., 'booking', 'cancelling', 'rescheduling', 'prescriptions', or 'general')\n"
        "6. Do not add any other text, explanations, or punctuation\n"
        "7. If the user mentions they want to meet, cancel, or reschedule with a doctor, consider their intent clear and forward to the router.\n"
        "8. Medicine / prescription / what did the doctor give me → prescriptions.\n"
        "9. Only ask clarifying questions if the intent is ambiguous or not actionable.\n"
        "Your response will be used directly by the system to route the user to the appropriate specialist."
    )
    return create_react_agent(model, tools, prompt=state_modifier)


# -------- Router + Specialized Agents --------
def build_general_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = []
    state_modifier = (
        f"You are a hospital assistant at {HospitalName}.\n"
        "- Handle greetings briefly in one short sentence.\n"
        "- Do NOT provide doctor suggestions and do NOT call any tools.\n"
        "- If unclear, ask ONE short question: book, cancel, reschedule, or prescriptions?\n"
        "- Do not say 'I understand' or restate the user's message.\n"
        "- NEVER collect phone, name, DOB, or symptoms yourself.\n"
        "- NEVER say the team will contact them, call them back, or that you will forward the request.\n"
        "- NEVER promise registration without booking. You only greet and clarify intent.\n"
        "- Once they want to book/cancel/reschedule/check medicines, reply with one short line "
        "like 'Okay, let's book that.' — the router sends them to the specialist next turn.\n"
        "- Plain spoken language only — never markdown.\n"
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
        "You are the Appointment Booking Agent for a live voice call.\n"
        "You complete booking, cancel, and reschedule yourself end-to-end. "
        "You are NOT a receptionist who forwards calls.\n"
        "Tools (ALL available — use them): lookup_patient, save_patient, list_doctors, "
        "book_appointment, cancel_appointment, reschedule_appointment.\n"
        "Never invent values. Use only what the user provides or what tools return.\n"
        "\n"
        "FORBIDDEN (critical):\n"
        "- NEVER say you lack cancel/reschedule/book/lookup tools — you have them.\n"
        "- NEVER say you cannot look up an existing appointment time — call lookup_patient "
        "and read active_appointment.time (and appointment_id) aloud.\n"
        "- NEVER say anyone will contact the patient later or that you forwarded the request.\n"
        "- NEVER invent register_patient, get_available_slots, get_appointment, or made-up "
        "availability lists — only speak nearest_times / alternate_doctors from a conflict tool result.\n"
        "- NEVER ask for date of birth, age, or government ID.\n"
        "- NEVER skip name confirmation for new bookings.\n"
        "- NEVER claim the patient is registered unless save_patient or book_appointment returned ok=true.\n"
        "\n"
        "ONE QUESTION PER TURN:\n"
        "- Ask only ONE short question each turn.\n"
        "- Infer fields already present — do not re-ask known info.\n"
        "\n"
        "EXISTING APPOINTMENT (critical):\n"
        "- lookup_patient returns active_appointment with appointment_id, doctor, and time. "
        "If they ask 'what time is my appointment?', call lookup_patient (phone) and speak that time. "
        "Do NOT start a new booking for that question.\n"
        "- If they already have an active_appointment and want a different day/time with the same doctor, "
        "call reschedule_appointment (appointment_id from active_appointment, or phone=...) "
        "with the new time after they confirm.\n"
        "- If they want to cancel then book a different doctor, call cancel_appointment first "
        "(id or phone), then book_appointment.\n"
        "- Prefer reschedule_appointment over cancel+book when only the time changes.\n"
        "- NEVER say you cannot cancel, reschedule, or look up appointments.\n"
        "\n"
        "STRICT NEW-BOOKING FLOW:\n"
        "1) Ask phone if unknown. Accumulate digits across short turns.\n"
        "2) Confirm phone aloud and get yes, THEN call lookup_patient(phone=...).\n"
        "3) Ask full name if needed.\n"
        "4) Confirm name aloud and get yes.\n"
        "5) After name is confirmed yes: for a returning patient call lookup_patient(phone=..., patient_name=...) "
        "again so active_appointment is fresh; for a new patient call save_patient(patient_name=..., phone=...). "
        "Then tell them their patient id briefly.\n"
        "6) Ask symptoms OR department (one question). Map tooth/dental pain → Dentistry; "
        "headache → General Medicine. Call list_doctors with that department "
        "(aliases like 'Dental' work).\n"
        "7) Ask preferred day and time early when possible (e.g. tomorrow at 10 AM).\n"
        "8) Offer 2-3 doctors via list_doctors(department=..., preferred_time=...). "
        "Clinic hours 9:00 AM–5:00 PM; lunch 2:00–3:00 PM is unavailable.\n"
        "   If preferred_time_unavailable=true OR every doctor has free_at_preferred_time=false, "
        "the department STILL has doctors — say that time is not available and offer "
        "nearest_times from the tool (e.g. 1:40 PM). Do NOT switch departments for a busy/lunch slot.\n"
        "   Only suggest another department when list_doctors returns count=0 for that specialty "
        "(truly no matching doctors), never because 2 PM is lunch.\n"
        "9) Confirm name + phone + doctor + time, then call book_appointment ONLY after yes.\n"
        "10) AVAILABILITY (critical): book_appointment checks the slot. If ok=false "
        "(time_conflict / doctor_unavailable / outside_clinic_hours), NEVER overwrite another patient. "
        "Offer nearest_times[0] first (same doctor). "
        "If they insist on the original time, offer alternate_doctors in the SAME department. "
        "Only leave the department if alternate_doctors is empty AND the patient agrees.\n"
        "11) After success, say the appointment id, patient id, and the concrete date/time returned.\n"
        "\n"
        "RETURNING PATIENT (lookup is_returning):\n"
        "- Mention active_appointment time/id when present (especially if they ask what time they have).\n"
        "- Mention last doctor if useful.\n"
        "- If they want a new time for the existing appointment → reschedule flow above.\n"
        "- Same doctor new booking only if no active appointment.\n"
        "- Different issue → other department via list_doctors.\n"
        "\n"
        "TOOLS:\n"
        "- Call tools via the normal tool interface only. Never print DSML/XML tool text.\n"
        "- Do not invent open slots — only speak nearest_times / alternate_doctors returned by "
        "book_appointment or reschedule_appointment on conflict.\n"
        "\n"
        "Plain spoken language only. Never markdown.\n"
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
        "You are the Cancellation Agent for a live voice call.\n"
        "Tools: lookup_patient, cancel_appointment, reschedule_appointment, book_appointment, list_doctors.\n"
        "- If appointment id is unknown, ask phone, confirm it, call lookup_patient, then use "
        "active_appointment.appointment_id or cancel_appointment(phone=...).\n"
        "- Confirm once before cancelling: 'Cancel appointment APT-0001. Should I go ahead?'\n"
        "- Call cancel_appointment ONLY after the user confirms. NEVER say you lack this tool.\n"
        "- If after cancel they want a new booking or a new time, you may book_appointment or "
        "reschedule_appointment yourself — do not claim you cannot.\n"
        "- After success, confirm briefly with the concrete result.\n"
        "- Plain spoken language only — never markdown.\n"
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
        "You are the Rescheduling Agent for a live voice call.\n"
        "Tools: lookup_patient, reschedule_appointment, cancel_appointment, book_appointment, list_doctors.\n"
        "- If appointment id is unknown: ask phone, confirm, lookup_patient(phone + name once confirmed), "
        "then read active_appointment.time if they ask when they are booked, or reschedule with "
        "appointment_id from active_appointment OR reschedule_appointment(phone=..., new_time=...).\n"
        "- NEVER say you lack a tool to look up appointment times — use lookup_patient.active_appointment.\n"
        "- Ask for the new day/time in one short question.\n"
        "- Confirm once, then call reschedule_appointment. NEVER say you lack this tool.\n"
        "- If ok=false time_conflict: never delete another booking. Offer nearest_times[0] "
        "with the same doctor first. If they insist on the original time, offer "
        "alternate_doctors from the tool result; on accept, cancel then book_appointment "
        "with the new doctor at that time.\n"
        "- If they want a different doctor instead, cancel then book, or use list_doctors then book.\n"
        "- Plain spoken language only — never markdown.\n"
    )
    return create_react_agent(model, tools, prompt=state_modifier)


def build_prescription_agent():
    model = _init_model(provider=os.getenv("CLI_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq")
    tools = [get_prescriptions]
    state_modifier = (
        "You are the Prescription Lookup Agent for a live voice call.\n"
        "Use ONLY the get_prescriptions tool. Never invent medicines.\n"
        "\n"
        "Patients often forget their patient id.\n"
        "- If they give an id (PAT-0001), use it.\n"
        "- Otherwise verify with BOTH name and phone. Collect ONE missing field per turn.\n"
        "- Prefer asking phone first, then ALWAYS confirm the number before calling the tool.\n"
        "- Then ask name, ALWAYS confirm the name, then call get_prescriptions with both.\n"
        "- NEVER ask for date of birth, age, or ID card.\n"
        "- If they offer a date of birth, say we only need phone and name, then continue.\n"
        "- Infer name/phone already spoken in earlier turns — do not re-ask, but still confirm once.\n"
        "- If the tool reports name_mismatch or asks for missing fields, follow that and ask one short question.\n"
        "\n"
        "When results return, speak briefly: medicine name, timing, and doctor.\n"
        "If none found, say so and ask them to recheck phone or name.\n"
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


def _sticky_route_from_history(
    messages: List[Any],
) -> Literal["general", "booking", "cancelling", "rescheduling", "prescriptions"] | None:
    """Keep mid-flow conversations on the specialist agent (esp. booking)."""
    recent = messages[-16:] if messages else []
    blob = " ".join(_message_plain(m).lower() for m in recent)
    last_user = ""
    for m in reversed(recent):
        if _message_role(m) in ("human", "user"):
            last_user = _message_plain(m).lower()
            break

    if any(k in last_user for k in ("cancel appointment", "cancel my", "i want to cancel")):
        return "cancelling"
    if any(k in last_user for k in ("reschedule", "move my appointment", "change the time")):
        return "rescheduling"
    if any(k in last_user for k in ("prescription", "medicine", "medication", "what did the doctor")):
        return "prescriptions"

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
        "patient id",
        "tomorrow",
        " am",
        " pm",
    )
    if any(k in blob for k in booking_markers):
        # Stay in booking unless user clearly switched topic to cancel/reschedule/rx above.
        if any(k in last_user for k in ("cancel", "reschedule", "prescription", "medicine")):
            return None
        return "booking"
    return None


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

        sticky = _sticky_route_from_history(all_messages)
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
        if "prescription" in response_text or "medicine" in response_text or "medication" in response_text:
            choice = "prescriptions"
        elif "cancelling" in response_text or "cancel" in response_text:
            choice = "cancelling"
        elif "rescheduling" in response_text or "reschedule" in response_text:
            choice = "rescheduling"
        elif "booking" in response_text or "book" in response_text:
            choice = "booking"
        else:
            choice = "general"

        # If LLM said general but user clearly asked to book, force booking.
        last_user = ""
        for m in reversed(all_messages or []):
            if _message_role(m) in ("human", "user"):
                last_user = _message_plain(m).lower()
                break
        if choice == "general" and any(k in last_user for k in ("book", "appointment")):
            choice = "booking"

        logging.info(f"route_decider choice: \n{choice}")
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