import os
import json
from dotenv import load_dotenv
import sys
from typing import  Any, List, TypedDict, Literal
load_dotenv()
import logging
import re
from datetime import datetime
from pprint import pformat
from langgraph.prebuilt import create_react_agent

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from Tools import list_doctors,book_appointment,cancel_appointment,reschedule_appointment
HospitalName="DBC"

logfile="app.log"
open(logfile,"w").close()
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def _init_model() -> Any:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Please export GROQ_API_KEY before running."
        )
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        groq_api_key=api_key,
    )


# -------- Router + Specialized Agents --------
def build_general_agent():
    model = _init_model()
    tools = []
    state_modifier = (
        f"You are a hospital assistant at {HospitalName}.\n"
        "- Handle general chit-chat briefly.\n"
        "- greet from hospital "
        "- Do NOT provide doctor suggestions and do NOT call any tools.\n"
        "- Your main job is to clarify the user's intent (booking, cancelling, rescheduling, or other) with short question if unclear.\n"
        "- don't provide available option try to clarify from small questions"
        "- Once intent is clear from their reply, the router will forward to the appropriate specialist."
    )
    return create_react_agent(model, tools, state_modifier=state_modifier)



def build_booking_agent():
    model = _init_model()
    tools = [book_appointment, list_doctors]
    state_modifier = (
        "You are the Appointment Booking Agent.\n"
        "- Your job is to BOOK an appointment using the tool.\n"
        "- Always ensure you have these fields before calling the tool: patient_name, doctor, time.\n"
        "- If the doctor is missing or not found, call list_doctors (optionally with a department or name query) to suggest valid choices from the directory; never invent names.\n"
        "- If doctor or preferred time are missing, ask for them (one short question at a time).\n"
        "- If patient_name is missing, ask for it as well.\n"
        "- The tool will validate the doctor name against the directory and refuse time conflicts.\n"
        "- Once you have all required fields, call book_appointment and then confirm the booking succinctly."
    )
    return create_react_agent(model, tools, state_modifier=state_modifier)


def build_cancellation_agent():
    model = _init_model()
    tools = [cancel_appointment]
    state_modifier = (
        "You are the Cancellation Agent.\n"
        "- Your job is to CANCEL an appointment.\n"
        "- Always ensure you have the appointment_id (e.g., APT-0001) before calling the tool.\n"
        "- If appointment_id is missing, ask for it concisely.\n"
        "- After a successful cancellation, confirm succinctly."
    )
    return create_react_agent(model, tools, state_modifier=state_modifier)


def build_reschedule_agent():
    model = _init_model()
    tools = [reschedule_appointment, list_doctors]
    state_modifier = (
        "You are the Rescheduling Agent.\n"
        "- Your job is to RESCHEDULE an existing appointment to a new time.\n"
        "- Always ensure you have appointment_id and new_time before calling the tool.\n"
        "- If either is missing, ask one concise clarifying question.\n"
        "- If the requested time is taken, inform the user and ask for another time.\n"
        "- If the user asks to change the doctor as part of rescheduling, call list_doctors (optionally filtered by department or name) to suggest valid doctors from the directory; never invent names."
    )
    return create_react_agent(model, tools, state_modifier=state_modifier)

class GraphState(TypedDict):
    messages: List[Any]


def build_graph():
    general_agent = build_general_agent()
    booking_agent = build_booking_agent()
    cancellation_agent = build_cancellation_agent()
    reschedule_agent = build_reschedule_agent()

    def _extract_last_user_text(messages: List[Any]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                return str(msg[1])
            role = getattr(msg, "type", getattr(msg, "role", None))
            if role == "human" or role == "user":
                content = getattr(msg, "content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def route_decider(state: GraphState) -> Literal["general", "booking", "cancelling", "rescheduling"]:
        logging.info(f"route_decider {pformat(state)}")
        model = _init_model()
        last_user = _extract_last_user_text(state.get("messages", []))
        system = (
            "You are a router that decides which specialist should handle the next turn.\n"
            "Return exactly one token: 'general', 'booking', 'cancelling', or 'rescheduling'.\n"
            "- 'booking' for booking or scheduling a new appointment.\n"
            "- 'cancelling' for cancellation requests.\n"
            "- 'rescheduling' when changing/moving an existing appointment's time.\n"
            "- Otherwise 'general'."
        )
        resp = model.invoke([
            ("system", system),
            ("user", f"Message: {last_user}")
        ])
        logging.info(f"route_decider resp {resp}")
        text = str(getattr(resp, "content", resp)).strip().lower()
        # Heuristic override to prefer specialized nodes when intent words appear in the raw user text
        raw = (last_user or "").lower()
        choice: str
        if ("cancel" in text or "cancelling" in text) or ("cancel" in raw or "delete appointment" in raw):
            choice = "cancelling"
        elif ("resched" in text or "re-sched" in text or "move" in text or "change time" in text) or ("resched" in raw or "reschedule" in raw or "move" in raw):
            choice = "rescheduling"
        elif ("book" in text or "appointment" in text or "schedule" in text) or ("book" in raw or "appointment" in raw or "schedule" in raw or "doctor" in raw):
            choice = "booking"
        else:
            choice = "general"
        logging.info(f"route_decider choise: {choice}")
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
        logging.info(f"Running node: general {pformat(state)}")
        prev = state.get("messages", [])
        result = general_agent.invoke({"messages": prev}, config={"recursion_limit": 5})
        logging.info(f"Running node: general result {pformat(result)}")
        msgs = result.get("messages", [])
        _append_chat_record("general", prev, msgs)
        return {"messages": msgs}

    def booking_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: booking {pformat(state)}")
        prev = state.get("messages", [])
        result = booking_agent.invoke({"messages": prev}, config={"recursion_limit": 5})
        msgs = result.get("messages", [])
        _append_chat_record("booking", prev, msgs)
        return {"messages": msgs}

    def cancelling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: cancelling {pformat(state)}")
        prev = state.get("messages", [])
        result = cancellation_agent.invoke({"messages": prev}, config={"recursion_limit": 5})
        msgs = result.get("messages", [])
        _append_chat_record("cancelling", prev, msgs)
        return {"messages": msgs}

    def rescheduling_node(state: GraphState) -> GraphState:
        logging.info(f"Running node: rescheduling {pformat(state)}")
        prev = state.get("messages", [])
        result = reschedule_agent.invoke({"messages": prev}, config={"recursion_limit": 5})
        msgs = result.get("messages", [])
        _append_chat_record("rescheduling", prev, msgs)
        return {"messages": msgs}

    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("booking", booking_node)
    graph.add_node("cancelling", cancelling_node)
    graph.add_node("rescheduling", rescheduling_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decider, {
        "general": "general",
        "booking": "booking",
        "cancelling": "cancelling",
        "rescheduling": "rescheduling",
    })
    # Let general immediately re-route in the same turn; specialists end the run
    # graph.add_edge("general", "router")
    graph.add_edge("general", END)
    graph.add_edge("booking", END)
    graph.add_edge("cancelling", END)
    graph.add_edge("rescheduling", END)
    return graph.compile()


def main():
    # Prepare per-session chat log file
    base_dir = os.path.dirname(__file__)
    chats_dir = os.path.join(base_dir, "chats")
    os.makedirs(chats_dir, exist_ok=True)
    session_name = datetime.now().strftime("chat-%Y%m%d-%H%M%S.jsonl")
    global CHAT_LOG_PATH
    CHAT_LOG_PATH = os.path.join(chats_dir, session_name)
    # create file with a header comment line
    try:
        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    except Exception as e:
        logging.error(f"Failed to initialize chat log file: {e}")

    app = build_graph()

    # Conversation state persists across turns
    messages: List[Any] = []

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        messages.append(("user", user_input))
        result = app.invoke({"messages": messages}, config={"recursion_limit": 5})
        messages = result.get("messages", messages)
        final = messages[-1] if messages else None
        content = getattr(final, "content", None) if final else ""
        print(content if isinstance(content, str) else str(final))
        return

    print("Hospital Agent ready. Type your request (Ctrl+C to exit).\n")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            messages.append(("user", user_input))
            result = app.invoke({"messages": messages}, config={"recursion_limit": 5})
            messages = result.get("messages", messages)
            final = messages[-1] if messages else None
            content = getattr(final, "content", None) if final else ""
            print(content if isinstance(content, str) else str(final))

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()