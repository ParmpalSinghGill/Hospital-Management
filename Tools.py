import logging
from tool_booking import book_appointment, cancel_appointment, reschedule_appointment
from tool_doctors import list_doctors
from tool_patient import lookup_patient, save_patient
from tool_prescriptions import get_prescriptions

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
