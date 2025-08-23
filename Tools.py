import json,logging
from typing import Optional
from langchain_core.tools import tool
from database import _load_doctors,_get_doctor_by_name,_load_all_appointments,_find_conflict,_get_next_appointment_id,_save_all_appointments

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
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
    logging.info("book_appointment %s %s %s",patient_name,doctor,time)
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
    logging.info("cancel_appointment %s",appointment_id)

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


@tool
def reschedule_appointment(appointment_id: str, new_time: str) -> str:
    """Reschedule an existing appointment to a new time, checking conflicts.

    Args:
        appointment_id: ID like APT-0001.
        new_time: Target time (free-text accepted; must exactly match stored format to avoid conflict misses).

    Returns:
        JSON string with ok, message, and updated appointment if successful.
    """
    logging.info("reschedule_appointment %s %s",appointment_id,new_time)
    appts = _load_all_appointments()
    # Find the appointment
    target = None
    for a in appts:
        if a.get("appointment_id") == appointment_id:
            target = a
            break
    if target is None:
        return json.dumps({"ok": False, "message": f"Appointment not found: {appointment_id}"})
    if target.get("status") == "CANCELLED":
        return json.dumps({"ok": False, "message": "Cannot reschedule a cancelled appointment."})

    doctor_name = target.get("doctor", "")
    conflict = _find_conflict(appts, doctor_name, new_time)
    if conflict and conflict.get("appointment_id") != appointment_id:
        return json.dumps({
            "ok": False,
            "message": f"Time conflict: {doctor_name} already has an appointment at '{new_time}'.",
            "conflict": conflict,
        })

    target["time"] = new_time
    _save_all_appointments(appts)
    return json.dumps({"ok": True, "message": "Appointment rescheduled", "appointment": target})



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
    logging.info("list_doctors %s %s",department, query)
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
