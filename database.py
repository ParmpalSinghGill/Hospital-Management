import json,os
from typing import  Optional,Any,List,Dict

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

