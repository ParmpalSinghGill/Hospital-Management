"""Prescriptions storage and lookup."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from db_core import _db, _next_id, _row_to_dict
from db_doctors import _get_doctor_by_id
from db_patients import _resolve_patient

def _load_prescriptions() -> List[Dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT prescription_id, patient_id, doctor_id, medicine_name, timing "
            "FROM prescriptions ORDER BY prescription_id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _save_prescriptions(prescriptions: List[Dict[str, Any]]) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM prescriptions")
        for r in prescriptions:
            conn.execute(
                "INSERT INTO prescriptions("
                "prescription_id, patient_id, doctor_id, medicine_name, timing"
                ") VALUES (?,?,?,?,?)",
                (
                    r.get("prescription_id", ""),
                    r.get("patient_id", ""),
                    r.get("doctor_id", ""),
                    r.get("medicine_name", ""),
                    r.get("timing", ""),
                ),
            )


def _get_next_prescription_id(prescriptions: Optional[List[Dict[str, Any]]] = None) -> str:
    if prescriptions is not None:
        return _next_id([p.get("prescription_id", "") for p in prescriptions], "RX")
    with _db() as conn:
        rows = conn.execute("SELECT prescription_id FROM prescriptions").fetchall()
    return _next_id([r["prescription_id"] for r in rows], "RX")


def _find_prescriptions(
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    phone: Optional[str] = None,
) -> Dict[str, Any]:
    """Find prescriptions after verifying the patient."""
    resolved = _resolve_patient(
        patient_id=patient_id,
        phone=phone,
        patient_name=patient_name,
        require_name_with_phone=not bool((patient_id or "").strip()),
    )
    if not resolved.get("ok") or not resolved.get("patient"):
        return {
            "ok": False,
            "message": resolved.get("message") or "Patient not found.",
            "prescriptions": [],
            "needs_name": resolved.get("needs_name", False),
            "needs_phone": resolved.get("needs_phone", False),
            "name_mismatch": resolved.get("name_mismatch", False),
        }

    patient = resolved["patient"]
    pid = patient.get("patient_id")
    rows = [r for r in _load_prescriptions() if r.get("patient_id") == pid]
    enriched = []
    for r in rows:
        item = dict(r)
        item["patient_name"] = patient.get("name", "")
        doctor = _get_doctor_by_id(str(r.get("doctor_id", "")))
        if doctor:
            item["doctor_name"] = doctor.get("name", "")
            item["department"] = doctor.get("department", "")
        enriched.append(item)

    if not enriched:
        return {
            "ok": False,
            "message": "No prescriptions on file for this patient.",
            "patient": patient,
            "prescriptions": [],
        }

    return {
        "ok": True,
        "message": f"Found {len(enriched)} prescription(s).",
        "patient": patient,
        "prescriptions": enriched,
        "count": len(enriched),
    }
