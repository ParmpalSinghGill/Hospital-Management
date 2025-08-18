import os
from dotenv import load_dotenv
import json
import sys
from typing import Dict, Any, List, Tuple, Optional
load_dotenv()
import logging
import re

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
# )

# ------------------------------
# JSON "databases" under ./dataset
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DOCTORS_DB = os.path.join(DATASET_DIR, "doctors.json")
APPOINTMENTS_DB = os.path.join(DATASET_DIR, "appointments.json")




def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _normalize_doctor_name(name: str) -> str:
    n = name.strip().lower()
    if n.startswith("dr. "):
        n = n[4:]
    elif n.startswith("dr "):
        n = n[3:]
    return n


def _load_doctors() -> List[Dict[str, str]]:
    db = _read_json(DOCTORS_DB, {"doctors": []})
    return db.get("doctors", [])


def _get_doctor_by_name(name: str) -> Optional[Dict[str, str]]:
    target = _normalize_doctor_name(name)
    for d in _load_doctors():
        if _normalize_doctor_name(d.get("name", "")) == target:
            return d
    return None


def _load_all_appointments() -> List[Dict[str, Any]]:
    db = _read_json(APPOINTMENTS_DB, {"appointments": []})
    return db.get("appointments", [])


def _save_all_appointments(appts: List[Dict[str, Any]]) -> None:
    _write_json(APPOINTMENTS_DB, {"appointments": appts})


def _extract_id_number(appointment_id: str) -> int:
    # APT-0001 -> 1
    try:
        return int(appointment_id.split("-")[1])
    except Exception:
        return 0


def _get_next_appointment_id(appts: List[Dict[str, Any]]) -> str:
    max_num = 0
    for a in appts:
        aid = a.get("appointment_id", "")
        max_num = max(max_num, _extract_id_number(aid))
    return f"APT-{(max_num + 1):04d}"


def _find_conflict(appts: List[Dict[str, Any]], doctor_name: str, time_str: str) -> Optional[Dict[str, Any]]:
    doc_norm = _normalize_doctor_name(doctor_name)
    time_norm = time_str.strip().lower()
    for a in appts:
        if a.get("status") == "CANCELLED":
            continue
        if _normalize_doctor_name(a.get("doctor", "")) == doc_norm and str(a.get("time", "")).strip().lower() == time_norm:
            return a
    return None


# ------------------------------
# Tools using JSON "databases"
# ------------------------------
@tool
def book_appointment(patient_name: str, doctor: str, time: str) -> str:
    """Book an appointment after validating doctor exists and time is free.

    Args:
        patient_name: The full name of the patient.
        doctor: The doctor's name the patient wants to see.
        time: The appointment time (free-text or ISO 8601).

    Returns:
        JSON string with result: ok, message, and appointment if created.
    """

    # Validate doctor
    doctor_row = _get_doctor_by_name(doctor)
    if not doctor_row:
        return json.dumps({
            "ok": False,
            "message": f"Doctor not found: {doctor}. Use the full name as in the directory.",
        })

    # Load and check conflicts
    appts = _load_all_appointments()
    conflict = _find_conflict(appts, doctor_row["name"], time)
    if conflict:
        return json.dumps({
            "ok": False,
            "message": f"Time conflict: {doctor_row['name']} already has an appointment at '{time}'.",
            "conflict": conflict,
        })

    # Create appointment
    appointment_id = _get_next_appointment_id(appts)
    new_appt = {
        "appointment_id": appointment_id,
        "patient_name": patient_name,
        "doctor": doctor_row["name"],  # store canonical name
        "department": doctor_row.get("department", ""),
        "time": time,
        "status": "BOOKED",
    }
    appts.append(new_appt)
    _save_all_appointments(appts)

    return json.dumps({
        "ok": True,
        "message": "Appointment booked",
        "appointment": new_appt,
    })


@tool
def cancel_appointment(appointment_id: str) -> str:
    """Cancel an existing appointment by its ID.

    Args:
        appointment_id: The appointment identifier (e.g., APT-0001).

    Returns:
        JSON string indicating whether the cancellation succeeded and the record (if any).
    """

    appts = _load_all_appointments()

    for a in appts:
        if a.get("appointment_id") == appointment_id:
            if a.get("status") == "CANCELLED":
                _save_all_appointments(appts)
                return json.dumps({
                    "ok": True,
                    "message": "Appointment already cancelled",
                    "appointment": a,
                })
            a["status"] = "CANCELLED"
            _save_all_appointments(appts)
            return json.dumps({
                "ok": True,
                "message": "Appointment cancelled",
                "appointment": a,
            })

    return json.dumps({
        "ok": False,
        "message": f"Appointment not found: {appointment_id}",
    })


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


# ------------------------------
# Tools using JSON "databases"
# ------------------------------
@tool
def list_doctors(department: Optional[str] = None, query: Optional[str] = None, limit: int = 10) -> str:
    """List doctors from the directory, optionally filtered by department or a name query.

    Args:
        department: Filter by department (case-insensitive; substring match).
        query: Filter by name (case-insensitive; matches without the 'Dr.' prefix).
        limit: Maximum number of results to return.

    Returns:
        JSON string with 'ok', 'count', and 'doctors' list.
    """
    docs = _load_doctors()
    dep = (department or "").strip().lower()
    q = (query or "").strip().lower()

    def _norm_name(n: str) -> str:
        n = n.strip().lower()
        if n.startswith("dr. "):
            n = n[4:]
        elif n.startswith("dr "):
            n = n[3:]
        return n

    if dep:
        docs = [d for d in docs if dep in (d.get("department", "").strip().lower())]

    if q:
        docs = [d for d in docs if q in _norm_name(d.get("name", ""))]

    docs = docs[: max(1, int(limit))]
    return json.dumps({"ok": True, "count": len(docs), "doctors": docs})

# -------- Router + Specialized Agents --------
def build_general_agent():
    print("build_general_agent call")
    model = _init_model()
    tools = [book_appointment, cancel_appointment, list_doctors]
    state_modifier = (
        "You are a hospital assistant.\n"
        "- Handle general chit-chat briefly.\n"
        "- Only call tools if the user explicitly asks to book or cancel an appointment.\n"
        "- If details are missing for booking or canceling, ask exactly one concise clarifying question.\n"
        "- When the user asks for a doctor suggestion or mentions a department, call list_doctors to suggest real doctors from the directory; never invent names.\n"
        "- When booking, the tool will validate the doctor against the directory and check for time conflicts."
    )
    return create_react_agent(model, tools, state_modifier=state_modifier)



def build_booking_agent():
    print("build_booking_agent call")
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

def detect_booking_intent(text: str) -> bool:
    t = text.lower()
    if "appointment" not in t and "appt" not in t:
        return False
    verbs = ["book", "schedule", "make", "set", "arrange", "fix", "reserve"]
    return any(v in t for v in verbs)


def run_turn(agent, prior_messages: List[Any], user_input: str) -> Tuple[List[Any], str]:
    input_messages = list(prior_messages) if prior_messages else []
    input_messages.append(("user", user_input))
    result = agent.invoke({"messages": input_messages}, config={"recursion_limit": 5})
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    content = getattr(final_message, "content", None) if final_message else ""
    content = content if isinstance(content, str) else (str(final_message) if final_message else "")
    return messages, content


def main():


    general_agent = build_general_agent()
    booking_agent = build_booking_agent()

    general_messages: List[Any] = []
    booking_messages: List[Any] = []
    active_mode: str = "general"  # "general" | "booking"

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        if detect_booking_intent(user_input):
            _, output = run_turn(booking_agent, booking_messages, user_input)
        else:
            _, output = run_turn(general_agent, general_messages, user_input)
        print(output)
        return

    print("Hospital Agent ready. Type your request (Ctrl+C to exit).")
    print("- Say 'book an appointment ...' to switch to the booking agent.\n")
    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if detect_booking_intent(user_input):
                active_mode = "booking"

            if active_mode == "booking":
                booking_messages, output = run_turn(booking_agent, booking_messages, user_input)
                print(output)
                if re.search(r"\bappointment booked\b", output.lower()):
                    active_mode = "general"
                    booking_messages = []
                continue

            general_messages, output = run_turn(general_agent, general_messages, user_input)
            print(output)

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()