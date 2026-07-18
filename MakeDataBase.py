import math
import os
import random
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Iterable

from database import (
    DB_PATH,
    init_db,
    clear_table,
    _load_doctors,
    _save_doctors,
    _load_patients,
    _save_patients,
    _load_all_appointments,
    _save_all_appointments,
    _load_prescriptions,
    _save_prescriptions,
    _normalize_doctor_name,
    _get_next_patient_id,
    _get_next_appointment_id,
    _get_next_prescription_id,
    ensure_default_lunch_breaks,
    is_doctor_unavailable,
    is_within_clinic_hours,
)

# ------------------------------
BASE_DIR = os.path.dirname(__file__)

FAKE_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Aadhya", "Diya", "Myra", "Ira", "Anika",
    "Sara", "Pari", "Navya", "Riya", "Kabir", "Rohan", "Neha", "Priya",
    "Rahul", "Amit", "Sneha", "Kavya", "Dev", "Meera",
]
FAKE_LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Mehta", "Reddy", "Nair",
    "Kapoor", "Joshi", "Verma", "Malhotra", "Iyer", "Chopra", "Das", "Khan",
    "Banerjee", "Pillai", "Rao", "Shah",
]
FAKE_STREETS = [
    "MG Road", "Park Street", "Lake View Lane", "Station Road", "Ring Road",
    "Gandhi Nagar", "Civil Lines", "Market Street", "Hill Crest", "Sunrise Avenue",
]
FAKE_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad",
]
FAKE_MEDICINES = [
    "MediClear", "HealFast", "PainAway", "VitaBoost", "CalmTabs", "FeverFix",
    "LungEase", "HeartTone", "SleepWell", "DigestAid", "AllergyGone", "BonePlus",
    "NerveCalm", "SugarBalance", "ImmuneGuard", "CoughStop", "SkinSoft", "EyeBright",
    "JointFlex", "BloodPure", "EnergyRise", "MoodLift", "ThroatCool", "NoseClear",
]
FAKE_TIMINGS = [
    "Once daily after breakfast",
    "Twice daily morning and night",
    "Three times daily after meals",
    "Every 8 hours",
    "At bedtime",
    "Morning only for 5 days",
    "With lunch and dinner",
    "Every 12 hours for 7 days",
    "As needed for pain",
    "Before breakfast daily",
]


def _default_doctors() -> List[Dict[str, str]]:
    """Canonical doctor directory used when SQLite has no doctors yet."""
    raw = [
                {"name": "Dr. Gregory House", "department": "Diagnostics"},
                {"name": "Dr. Allison Cameron", "department": "Diagnostics"},
                {"name": "Dr. Robert Chase", "department": "Diagnostics"},
                {"name": "Dr. Meredith Grey", "department": "General Surgery"},
                {"name": "Dr. Miranda Bailey", "department": "General Surgery"},
                {"name": "Dr. Richard Webber", "department": "General Surgery"},
                {"name": "Dr. Derek Shepherd", "department": "Neurosurgery"},
                {"name": "Dr. Amelia Shepherd", "department": "Neurosurgery"},
                {"name": "Dr. Thomas Avery", "department": "Neurosurgery"},
                {"name": "Dr. Cristina Yang", "department": "Cardiothoracic Surgery"},
                {"name": "Dr. Preston Burke", "department": "Cardiothoracic Surgery"},
                {"name": "Dr. Erica Hahn", "department": "Cardiothoracic Surgery"},
                {"name": "Dr. James Wilson", "department": "Oncology"},
                {"name": "Dr. April Kepner", "department": "Oncology"},
                {"name": "Dr. Leah Murphy", "department": "Oncology"},
                {"name": "Dr. Lisa Cuddy", "department": "Endocrinology"},
                {"name": "Dr. Maya Santos", "department": "Endocrinology"},
                {"name": "Dr. Nina Kapoor", "department": "Endocrinology"},
                {"name": "Dr. Arizona Robbins", "department": "Pediatric Surgery"},
                {"name": "Dr. Alex Karev", "department": "Pediatric Surgery"},
                {"name": "Dr. Sofia Rossi", "department": "Pediatric Surgery"},
                {"name": "Dr. Mark Sloan", "department": "Plastic Surgery"},
                {"name": "Dr. Jackson Avery", "department": "Plastic Surgery"},
                {"name": "Dr. Jo Wilson", "department": "Plastic Surgery"},
                {"name": "Dr. Owen Hunt", "department": "Trauma Surgery"},
                {"name": "Dr. Teddy Altman", "department": "Trauma Surgery"},
                {"name": "Dr. Megan Hunt", "department": "Trauma Surgery"},
                {"name": "Dr. Mark Hunt", "department": "Orthopedics"},
                {"name": "Dr. Owen Sloan", "department": "Orthopedics"},
                {"name": "Dr. Callie Torres", "department": "Orthopedics"},
                {"name": "Dr. Eric Foreman", "department": "General Medicine"},
                {"name": "Dr. Laura Chen", "department": "General Medicine"},
                {"name": "Dr. Sam Okoro", "department": "General Medicine"},
                {"name": "Dr. Maggie Pierce", "department": "Cardiology"},
                {"name": "Dr. Andrew DeLuca", "department": "Cardiology"},
                {"name": "Dr. Helen Park", "department": "Cardiology"},
                {"name": "Dr. Addison Montgomery", "department": "Dermatology"},
                {"name": "Dr. Priya Nair", "department": "Dermatology"},
                {"name": "Dr. Jordan Blake", "department": "Dermatology"},
                {"name": "Dr. Izzie Stevens", "department": "ENT"},
                {"name": "Dr. Marcus Lee", "department": "ENT"},
                {"name": "Dr. Hannah Cole", "department": "ENT"},
                {"name": "Dr. George O'Malley", "department": "Ophthalmology"},
                {"name": "Dr. Elena Ruiz", "department": "Ophthalmology"},
                {"name": "Dr. Victor Haas", "department": "Ophthalmology"},
                {"name": "Dr. Addison Forbes", "department": "Gynecology"},
                {"name": "Dr. Naomi Bennett", "department": "Gynecology"},
                {"name": "Dr. Carina DeLuca", "department": "Gynecology"},
                {"name": "Dr. Charlotte King", "department": "Pediatrics"},
                {"name": "Dr. Aria Patel", "department": "Pediatrics"},
                {"name": "Dr. Ben Warren", "department": "Pediatrics"},
                {"name": "Dr. Violet Turner", "department": "Psychiatry"},
                {"name": "Dr. Sheldon Cooper", "department": "Psychiatry"},
                {"name": "Dr. Maya Rendell", "department": "Psychiatry"},
                {"name": "Dr. Ethan Baker", "department": "Emergency Medicine"},
                {"name": "Dr. Connie Beaufort", "department": "Emergency Medicine"},
                {"name": "Dr. Noah Greer", "department": "Emergency Medicine"},
                {"name": "Dr. Amelia Reid", "department": "Family Medicine"},
                {"name": "Dr. Carlos Mendez", "department": "Family Medicine"},
                {"name": "Dr. Fatima Rahman", "department": "Family Medicine"},
                {"name": "Dr. Denny Duquette", "department": "Gastroenterology"},
                {"name": "Dr. Olivia Grant", "department": "Gastroenterology"},
                {"name": "Dr. Ryan Walsh", "department": "Gastroenterology"},
                {"name": "Dr. Lexie Grey", "department": "Pulmonology"},
                {"name": "Dr. Stephanie Edwards", "department": "Pulmonology"},
                {"name": "Dr. Nathan Riggs", "department": "Pulmonology"},
                {"name": "Dr. Shane Ross", "department": "Nephrology"},
                {"name": "Dr. Aisha Khan", "department": "Nephrology"},
                {"name": "Dr. Paul Kim", "department": "Nephrology"},
                {"name": "Dr. Nico Kim", "department": "Urology"},
                {"name": "Dr. Gabriel Garcia", "department": "Urology"},
                {"name": "Dr. Emily Frost", "department": "Urology"},
                {"name": "Dr. Adele Webber", "department": "Neurology"},
                {"name": "Dr. Kai Bartley", "department": "Neurology"},
                {"name": "Dr. Mira Solis", "department": "Neurology"},
                {"name": "Dr. Tom Koracick", "department": "Rheumatology"},
                {"name": "Dr. Heather Brooks", "department": "Rheumatology"},
                {"name": "Dr. Daniel Pierce", "department": "Rheumatology"},
                {"name": "Dr. Catherine Fox", "department": "Infectious Disease"},
                {"name": "Dr. Jules Millin", "department": "Infectious Disease"},
                {"name": "Dr. Omar Farid", "department": "Infectious Disease"},
                {"name": "Dr. Link Adams", "department": "Allergy & Immunology"},
                {"name": "Dr. Sabrina Vazquez", "department": "Allergy & Immunology"},
                {"name": "Dr. Theo Rutherford", "department": "Allergy & Immunology"},
                {"name": "Dr. Penelope Alvarez", "department": "Radiology"},
                {"name": "Dr. Quinn Harper", "department": "Radiology"},
                {"name": "Dr. Yusuf Ali", "department": "Radiology"},
                {"name": "Dr. Cormac Hayes", "department": "Anesthesiology"},
                {"name": "Dr. Rose Rivera", "department": "Anesthesiology"},
                {"name": "Dr. Ian Fletcher", "department": "Anesthesiology"},
                {"name": "Dr. Taryn Helm", "department": "Physical Medicine & Rehabilitation"},
                {"name": "Dr. Levi Schmitt", "department": "Physical Medicine & Rehabilitation"},
                {"name": "Dr. Nora Lind", "department": "Physical Medicine & Rehabilitation"},
                {"name": "Dr. Jo Karev", "department": "Dentistry"},
                {"name": "Dr. Max Goodwin", "department": "Dentistry"},
                {"name": "Dr. Claire Yoon", "department": "Dentistry"},
                {"name": "Dr. Simone Griffith", "department": "Neonatology"},
                {"name": "Dr. Mika Yasuda", "department": "Neonatology"},
                {"name": "Dr. Lucas Adams", "department": "Neonatology"},
                {"name": "Dr. Ellis Grey", "department": "Geriatrics"},
                {"name": "Dr. Harold Finch", "department": "Geriatrics"},
                {"name": "Dr. Ruth Keller", "department": "Geriatrics"},
            ]
    doctors = []
    for i, doc in enumerate(raw):
        doctors.append({
            "doctor_id": f"DOC-{(i + 1):04d}",
            "name": doc["name"],
            "department": doc["department"],
        })
    return doctors


def _ensure_dataset_seeded() -> None:
    """Ensure SQLite exists; seed doctors if the table is empty."""
    init_db()
    if not _load_doctors():
        _save_doctors(_default_doctors())
    ensure_default_lunch_breaks()


def _bookable_slots_for_doctor(
    doctor_id: str,
    slots: List[str],
    taken: set,
) -> List[str]:
    """Clinic slots that are free, inside hours, and not in an unavailable block."""
    free = []
    for s in slots:
        if s.strip().lower() in taken:
            continue
        if not is_within_clinic_hours(s):
            continue
        if is_doctor_unavailable(doctor_id, s):
            continue
        free.append(s)
    return free


def _slots_to_fill(free_count: int, per_doctor: int, fill_ratio: float) -> int:
    """Book at least fill_ratio of free slots (default 50%), or per_doctor if higher."""
    if free_count <= 0:
        return 0
    min_half = max(1, math.ceil(free_count * max(0.0, min(1.0, fill_ratio))))
    return min(free_count, max(min_half, int(per_doctor or 0)))


def clear_appointments() -> int:
    """Clear all appointments. Returns rows removed."""
    init_db()
    return clear_table("appointments")


def clear_patients() -> int:
    init_db()
    return clear_table("patients")


def clear_prescriptions() -> int:
    init_db()
    return clear_table("prescriptions")


def clear_all(reseed_doctors: bool = True) -> dict:
    """Clear appointments, patients, prescriptions, and doctors. Optionally re-seed doctors."""
    init_db()
    stats = {
        "appointments_removed": clear_table("appointments"),
        "patients_removed": clear_table("patients"),
        "prescriptions_removed": clear_table("prescriptions"),
        "doctors_removed": clear_table("doctors"),
        "reseeded": False,
        "db_path": DB_PATH,
    }
    if reseed_doctors:
        _save_doctors(_default_doctors())
        stats["reseeded"] = True
    return stats


# -------- Time/slot helpers --------
ISO_FMT = "%Y-%m-%d %H:%M"


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def _generate_slots_for_date(
    d: date, start_hm: str = "09:00", end_hm: str = "17:00", slot_minutes: int = 10
) -> List[str]:
    sh, sm = [int(x) for x in start_hm.split(":")]
    eh, em = [int(x) for x in end_hm.split(":")]
    start_dt = datetime.combine(d, time(sh, sm))
    end_dt = datetime.combine(d, time(eh, em))
    slots = []
    cur = start_dt
    delta = timedelta(minutes=slot_minutes)
    while cur + delta <= end_dt:
        slots.append(cur.strftime(ISO_FMT))
        cur += delta
    return slots


def _random_patient(rng: random.Random, patient_id: str) -> Dict[str, Any]:
    name = f"{rng.choice(FAKE_FIRST_NAMES)} {rng.choice(FAKE_LAST_NAMES)}"
    phone = f"+91{rng.randint(7000000000, 9999999999)}"
    address = (
        f"{rng.randint(1, 999)}, {rng.choice(FAKE_STREETS)}, "
        f"{rng.choice(FAKE_CITIES)} - {rng.randint(100000, 999999)}"
    )
    return {
        "patient_id": patient_id,
        "name": name,
        "phone": phone,
        "address": address,
    }


def seed_patients(count: int = 80, rng_seed: Optional[int] = None) -> int:
    """Create random patients if the table is empty or extend up to `count`."""
    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)
    patients = _load_patients()
    created = 0
    while len(patients) < count:
        pid = _get_next_patient_id(patients)
        patients.append(_random_patient(rng, pid))
        created += 1
    _save_patients(patients)
    return created


def seed_prescriptions(
    per_patient: int = 2,
    rng_seed: Optional[int] = None,
) -> int:
    """Seed random fake prescriptions. Guarantees every patient has at least one."""
    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)
    doctors = _load_doctors()
    patients = _load_patients()
    if not doctors or not patients:
        return 0

    prescriptions = _load_prescriptions()
    by_patient: Dict[str, int] = {}
    existing_pairs = set()
    for p in prescriptions:
        pid = p.get("patient_id")
        by_patient[pid] = by_patient.get(pid, 0) + 1
        existing_pairs.add((pid, p.get("medicine_name"), p.get("timing")))

    created = 0
    target = max(1, per_patient)

    for patient in patients:
        pid = patient["patient_id"]
        while by_patient.get(pid, 0) < target:
            doctor = rng.choice(doctors)
            # Retry a few times to avoid duplicate medicine+timing pairs
            added = False
            for _ in range(20):
                medicine = rng.choice(FAKE_MEDICINES)
                timing = rng.choice(FAKE_TIMINGS)
                key = (pid, medicine, timing)
                if key in existing_pairs:
                    continue
                existing_pairs.add(key)
                rx = {
                    "prescription_id": _get_next_prescription_id(prescriptions),
                    "patient_id": pid,
                    "doctor_id": doctor.get("doctor_id"),
                    "medicine_name": medicine,
                    "timing": timing,
                }
                prescriptions.append(rx)
                by_patient[pid] = by_patient.get(pid, 0) + 1
                created += 1
                added = True
                break
            if not added:
                # Force a unique timing suffix so every patient still gets a row
                medicine = rng.choice(FAKE_MEDICINES)
                timing = f"{rng.choice(FAKE_TIMINGS)} (ref {by_patient.get(pid, 0) + 1})"
                key = (pid, medicine, timing)
                existing_pairs.add(key)
                prescriptions.append({
                    "prescription_id": _get_next_prescription_id(prescriptions),
                    "patient_id": pid,
                    "doctor_id": doctor.get("doctor_id"),
                    "medicine_name": medicine,
                    "timing": timing,
                })
                by_patient[pid] = by_patient.get(pid, 0) + 1
                created += 1

    _save_prescriptions(prescriptions)
    return created


# -------- Random seeding --------
def _available_patients(
    patients: List[Dict[str, Any]],
    appts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    booked = {
        str(a.get("patient_id"))
        for a in appts
        if a.get("patient_id") and a.get("status") != "CANCELLED"
    }
    return [p for p in patients if str(p.get("patient_id")) not in booked]


def _take_free_patient(
    rng: random.Random,
    patients: List[Dict[str, Any]],
    appts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick a patient with no active appointment; create one if the pool is empty."""
    free = _available_patients(patients, appts)
    if free:
        patient = rng.choice(free)
        return patient
    # Create a fresh patient so we never reuse someone who already has a booking
    pid = _get_next_patient_id(patients)
    patient = _random_patient(rng, pid)
    patients.append(patient)
    _save_patients(patients)
    return patient


def seed_random_appointments_for_dates(
    dates: List[str],
    per_doctor: int = 6,
    start_hm: str = "09:00",
    end_hm: str = "17:00",
    slot_minutes: int = 10,
    rng_seed: Optional[int] = None,
    fill_ratio: float = 0.5,
) -> int:
    """
    Seed random BOOKED appointments for the given YYYY-MM-DD dates.
    - Avoids conflicts (per doctor + time).
    - Each patient gets at most ONE active appointment.
    - Returns count of new appointments created.
    """
    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)

    doctors = _load_doctors()
    patients = _load_patients()
    if not patients:
        seed_patients(count=80, rng_seed=rng_seed)
        patients = _load_patients()

    appts = _load_all_appointments()
    created = 0

    for ds in dates:
        d = _parse_date(ds)
        slots = _generate_slots_for_date(d, start_hm, end_hm, slot_minutes)
        for doc in doctors:
            taken = {
                str(a.get("time", "")).strip().lower()
                for a in appts
                if _normalize_doctor_name(a.get("doctor", "")) == _normalize_doctor_name(doc["name"])
                and a.get("status") != "CANCELLED"
            }
            free_slots = _bookable_slots_for_doctor(str(doc.get("doctor_id") or ""), slots, taken)
            if not free_slots:
                continue
            k = _slots_to_fill(len(free_slots), per_doctor, fill_ratio)
            chosen = rng.sample(free_slots, k=k)

            for ts in chosen:
                patient = _take_free_patient(rng, patients, appts)
                if patient is None:
                    continue
                patients = _load_patients()  # refresh after possible create
                aid = _get_next_appointment_id(appts)
                appt = {
                    "appointment_id": aid,
                    "patient_id": patient["patient_id"],
                    "doctor_id": doc.get("doctor_id"),
                    "doctor": doc["name"],
                    "department": doc.get("department", ""),
                    "time": ts,
                    "status": "BOOKED",
                }
                appts.append(appt)
                created += 1

    _save_all_appointments(appts)
    return created


def seed_random_appointments_range(
    start_date: str,
    end_date: str,
    per_doctor: int = 6,
    start_hm: str = "09:00",
    end_hm: str = "17:00",
    slot_minutes: int = 10,
    rng_seed: Optional[int] = None,
    fill_ratio: float = 0.5,
) -> int:
    """Seed random appointments for an inclusive date range [start_date, end_date], YYYY-MM-DD."""
    s = _parse_date(start_date)
    e = _parse_date(end_date)
    dates = [d.strftime("%Y-%m-%d") for d in _daterange(s, e)]
    return seed_random_appointments_for_dates(
        dates,
        per_doctor=per_doctor,
        start_hm=start_hm,
        end_hm=end_hm,
        slot_minutes=slot_minutes,
        rng_seed=rng_seed,
        fill_ratio=fill_ratio,
    )


def seed_random_appointments_days(
    days: int,
    per_doctor: int = 6,
    start_hm: str = "09:00",
    end_hm: str = "17:00",
    slot_minutes: int = 10,
    rng_seed: Optional[int] = None,
    fill_ratio: float = 0.5,
) -> int:
    """
    Seed random appointments for `days` days including today.
    - For today, only slots strictly after current time are considered.
    - Each patient gets at most ONE active appointment.
    """
    if days <= 0:
        return 0

    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)

    doctors = _load_doctors()
    patients = _load_patients()
    if not patients:
        seed_patients(count=80, rng_seed=rng_seed)
        patients = _load_patients()

    appts = _load_all_appointments()
    created = 0

    now = datetime.now()
    today = now.date()

    for i in range(days):
        d = today + timedelta(days=i)
        slots = _generate_slots_for_date(d, start_hm, end_hm, slot_minutes)
        if d == today:
            slots = [s for s in slots if datetime.strptime(s, ISO_FMT) > now]
        if not slots:
            continue

        for doc in doctors:
            taken = {
                str(a.get("time", "")).strip().lower()
                for a in appts
                if _normalize_doctor_name(a.get("doctor", "")) == _normalize_doctor_name(doc["name"])
                and a.get("status") != "CANCELLED"
            }
            free_slots = _bookable_slots_for_doctor(str(doc.get("doctor_id") or ""), slots, taken)
            if not free_slots:
                continue

            k = _slots_to_fill(len(free_slots), per_doctor, fill_ratio)
            chosen = rng.sample(free_slots, k=k)

            for ts in chosen:
                patient = _take_free_patient(rng, patients, appts)
                if patient is None:
                    continue
                patients = _load_patients()
                aid = _get_next_appointment_id(appts)
                appt = {
                    "appointment_id": aid,
                    "patient_id": patient["patient_id"],
                    "doctor_id": doc.get("doctor_id"),
                    "doctor": doc["name"],
                    "department": doc.get("department", ""),
                    "time": ts,
                    "status": "BOOKED",
                }
                appts.append(appt)
                created += 1

    _save_all_appointments(appts)
    return created


def seed_booking_chats(limit: Optional[int] = None) -> int:
    """Create a sample booking chat transcript for each active appointment."""
    from conversation_log import record_booking_chat

    patients_by_id = {p.get("patient_id"): p for p in _load_patients()}
    created = 0
    for a in _load_all_appointments():
        if a.get("status") == "CANCELLED":
            continue
        patient = patients_by_id.get(a.get("patient_id"))
        if not patient:
            continue
        record_booking_chat(patient=patient, appointment=a)
        created += 1
        if limit is not None and created >= limit:
            break
    return created


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed random appointments / patients / prescriptions.")
    parser.add_argument("--clear-appointments", action="store_true", help="Clear appointments table.")
    parser.add_argument("--clear-patients", action="store_true", help="Clear patients table.")
    parser.add_argument("--clear-prescriptions", action="store_true", help="Clear prescriptions table.")
    parser.add_argument("--clear-all", action="store_true", help="Clear SQLite tables and re-seed doctors.")
    parser.add_argument("--no-reseed", action="store_true", help="With --clear-all, do not re-seed doctors.")
    parser.add_argument("--patients", type=int, default=80, help="Ensure at least N patients exist (default 80).")
    parser.add_argument("--prescriptions-per-patient", type=int, default=2, help="Fake Rx rows per patient.")
    parser.add_argument("--skip-prescriptions", action="store_true", help="Do not seed prescriptions.")
    parser.add_argument("--skip-chats", action="store_true", help="Do not seed booking chat transcripts.")
    parser.add_argument("--days", type=int, default=3, help="Number of days including today to seed.")
    parser.add_argument("date", nargs="*", help="Dates YYYY-MM-DD. If --range, provide START END.")
    parser.add_argument("--range", action="store_true", help="Interpret provided dates as inclusive range.")
    parser.add_argument("--per-doctor", type=int, default=0, help="Minimum slots per doctor (0 = use fill-ratio only).")
    parser.add_argument("--fill-ratio", type=float, default=0.5, help="Fill at least this fraction of free slots (default 0.5).")
    parser.add_argument("--start", default="09:00", help="Day start time HH:MM (default 09:00).")
    parser.add_argument("--end", default="17:00", help="Day end time HH:MM (default 17:00).")
    parser.add_argument("--slot-minutes", type=int, default=10, help="Slot duration (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()
    _ensure_dataset_seeded()

    if args.clear_all:
        stats = clear_all(reseed_doctors=(not args.no_reseed))
        print(
            f"Cleared database: removed {stats['appointments_removed']} appointments, "
            f"{stats['patients_removed']} patients, {stats['prescriptions_removed']} prescriptions. "
            f"reseeded_doctors={stats['reseeded']}"
        )

    if args.clear_appointments:
        n = clear_appointments()
        print(f"Cleared appointments; removed {n} appointments.")

    if args.clear_patients:
        n = clear_patients()
        print(f"Cleared patients; removed {n} patients.")

    if args.clear_prescriptions:
        n = clear_prescriptions()
        print(f"Cleared prescriptions; removed {n} prescriptions.")

    created_patients = seed_patients(count=args.patients, rng_seed=args.seed)
    print(f"Patients ready (created {created_patients}, target >= {args.patients}).")

    if args.days is not None and not args.date:
        if args.days <= 0:
            print("No appointments created: --days must be >= 1.")
            created = 0
        else:
            created = seed_random_appointments_days(
                args.days,
                per_doctor=args.per_doctor,
                fill_ratio=args.fill_ratio,
                start_hm=args.start,
                end_hm=args.end,
                slot_minutes=args.slot_minutes,
                rng_seed=args.seed,
            )
    elif args.range:
        if len(args.date) != 2:
            raise SystemExit("Provide exactly two dates for --range: START END")
        created = seed_random_appointments_range(
            args.date[0],
            args.date[1],
            per_doctor=args.per_doctor,
                fill_ratio=args.fill_ratio,
            start_hm=args.start,
            end_hm=args.end,
            slot_minutes=args.slot_minutes,
            rng_seed=args.seed,
        )
    elif args.date:
        created = seed_random_appointments_for_dates(
            args.date,
            per_doctor=args.per_doctor,
                fill_ratio=args.fill_ratio,
            start_hm=args.start,
            end_hm=args.end,
            slot_minutes=args.slot_minutes,
            rng_seed=args.seed,
        )
    else:
        created = 0

    print(f"Created {created} random appointments (max 1 per patient).")
    print(f"Patients now: {len(_load_patients())}.")

    if not args.skip_prescriptions:
        created_rx = seed_prescriptions(
            per_patient=args.prescriptions_per_patient,
            rng_seed=args.seed,
        )
        print(f"Created {created_rx} random prescriptions.")

    if not args.skip_chats:
        created_chats = seed_booking_chats()
        print(f"Created {created_chats} booking chat transcripts.")
