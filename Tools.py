import logging

import security_config as _sc
from phi_redact import install_log_redaction as _install_log_redaction
from tool_booking import book_appointment, cancel_appointment, reschedule_appointment
from tool_doctors import list_doctors
from tool_patient import lookup_patient, save_patient
from tool_prescriptions import get_prescriptions

logging.basicConfig(
    filename=str(_sc.app_log_path()),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
_install_log_redaction()
