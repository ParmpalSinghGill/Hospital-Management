import json
import logging
from typing import Optional

from langchain_core.tools import tool

from audit_log import audit, subject_ref
from database import _find_prescriptions
from phi_redact import mask_phone


@tool
def get_prescriptions(
    patient_name: Optional[str] = None,
    phone: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> str:
    """Look up medicines prescribed for a patient. Requires BOTH phone and full name
    (a patient id alone is not enough) — medication data is sensitive."""
    logging.info("get_prescriptions patient_id=%s phone=%s", patient_id, mask_phone(phone))

    ph = (phone or "").strip()
    name = (patient_name or "").strip()
    if not ph or not name:
        return json.dumps({
            "ok": False,
            "message": "Ask for the phone number and the full name to verify before sharing prescriptions.",
            "needs_phone": not bool(ph),
            "needs_name": not bool(name),
            "prescriptions": [],
        })

    result = _find_prescriptions(
        patient_id=(patient_id or "").strip() or None,
        patient_name=name,
        phone=ph,
    )
    if result.get("name_mismatch"):
        result = {
            "ok": False,
            "message": "The name given does not match our record for this phone. Ask them to repeat their full name.",
            "name_mismatch": True,
            "needs_name": True,
            "prescriptions": [],
        }
        audit("tool.get_prescriptions", actor="agent", outcome="name_mismatch", subject=subject_ref(phone=ph))
    elif result.get("patient"):
        audit("tool.get_prescriptions", actor="agent",
              subject=subject_ref(patient_id=str((result.get("patient") or {}).get("patient_id") or "")),
              detail={"count": len(result.get("prescriptions") or [])})
    return json.dumps(result)
