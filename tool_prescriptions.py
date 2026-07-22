import json
import logging
from typing import Optional

from langchain_core.tools import tool

from database import _find_prescriptions


@tool
def get_prescriptions(
    patient_name: Optional[str] = None,
    phone: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> str:
    """Look up medicines prescribed for a patient."""
    logging.info("get_prescriptions patient_id=%s phone=%s name=%s", patient_id, phone, patient_name)

    if not any([(patient_id or "").strip(), (phone or "").strip(), (patient_name or "").strip()]):
        return json.dumps({
            "ok": False,
            "message": "Ask for patient id if they know it, otherwise start with their phone number.",
            "needs_phone": True,
        })

    result = _find_prescriptions(
        patient_id=(patient_id or "").strip() or None,
        patient_name=(patient_name or "").strip() or None,
        phone=(phone or "").strip() or None,
    )
    return json.dumps(result)
