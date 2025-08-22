import os, json, random
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Iterable

# ------------------------------
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DOCTORS_DB = os.path.join(DATASET_DIR, "doctors.json")
APPOINTMENTS_DB = os.path.join(DATASET_DIR, "appointments.json")


def _ensure_dataset_seeded() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)

    if not os.path.exists(DOCTORS_DB):
        doctors_seed = {
            "doctors": [
                {"name": "Dr. Gregory House", "department": "Diagnostics"},
                {"name": "Dr. Meredith Grey", "department": "General Surgery"},
                {"name": "Dr. Derek Shepherd", "department": "Neurosurgery"},
                {"name": "Dr. Cristina Yang", "department": "Cardiothoracic Surgery"},
                {"name": "Dr. James Wilson", "department": "Oncology"},
                {"name": "Dr. Lisa Cuddy", "department": "Endocrinology"},
                {"name": "Dr. Miranda Bailey", "department": "General Surgery"},
                {"name": "Dr. Arizona Robbins", "department": "Pediatric Surgery"},
                {"name": "Dr. Mark Sloan", "department": "Plastic Surgery"},
                {"name": "Dr. Owen Hunt", "department": "Trauma Surgery"},
                {"name": "Dr. Mark Hunt", "department": "Orthopedics"},
                {"name": "Dr. Owen Sloan", "department": "Orthopedics"},
            ]
        }
        with open(DOCTORS_DB, "w", encoding="utf-8") as f:
            json.dump(doctors_seed, f, indent=2)

    if not os.path.exists(APPOINTMENTS_DB):
        with open(APPOINTMENTS_DB, "w", encoding="utf-8") as f:
            json.dump({"appointments": []}, f, indent=2)


# -------- JSON helpers --------
def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clear_appointments() -> int:
    """Reset appointments.json to an empty list. Returns the number of entries removed."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    old = _read_json(APPOINTMENTS_DB, {"appointments": []})
    old_count = len(old.get("appointments", []))
    _write_json(APPOINTMENTS_DB, {"appointments": []})
    return old_count


def clear_all(reseed_doctors: bool = True) -> dict:
    """Delete appointments and doctors files. Optionally re-seed doctors.

    Returns a dict with counts of removed appointments and whether files existed.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    stats = {"appointments_removed": 0, "appointments_file": False, "doctors_file": False, "reseeded": False}

    old_appts = _read_json(APPOINTMENTS_DB, {"appointments": []})
    stats["appointments_removed"] = len(old_appts.get("appointments", []))
    try:
        os.remove(APPOINTMENTS_DB)
        stats["appointments_file"] = True
    except FileNotFoundError:
        pass
    try:
        os.remove(DOCTORS_DB)
        stats["doctors_file"] = True
    except FileNotFoundError:
        pass

    if reseed_doctors:
        _ensure_dataset_seeded()
        stats["reseeded"] = True

    return stats


def _load_doctors() -> List[Dict[str, str]]:
    db = _read_json(DOCTORS_DB, {"doctors": []})
    return db.get("doctors", [])


def _load_all_appointments() -> List[Dict[str, Any]]:
    db = _read_json(APPOINTMENTS_DB, {"appointments": []})
    return db.get("appointments", [])


def _save_all_appointments(appts: List[Dict[str, Any]]) -> None:
    _write_json(APPOINTMENTS_DB, {"appointments": appts})


def _extract_id_number(appointment_id: str) -> int:
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


def _normalize_doctor_name(name: str) -> str:
    n = name.strip().lower()
    if n.startswith("dr. "):
        n = n[4:]
    elif n.startswith("dr "):
        n = n[3:]
    return n


def _normalize_time_to_minute(s: str) -> Optional[str]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    fmts = ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None


def _find_conflict(appts: List[Dict[str, Any]], doctor_name: str, time_str: str) -> Optional[Dict[str, Any]]:
    doc_norm = _normalize_doctor_name(doctor_name)
    norm_req = _normalize_time_to_minute(time_str)
    if not norm_req:
        return None
    for a in appts:
        if a.get("status") == "CANCELLED":
            continue
        if _normalize_doctor_name(a.get("doctor", "")) != doc_norm:
            continue
        a_time = _normalize_time_to_minute(str(a.get("time", "")))
        if a_time and a_time == norm_req:
            return a
    return None


# -------- Random seeding --------
def seed_random_appointments_for_dates(
    dates: List[str],
    per_doctor: int = 6,
    start_hm: str = "09:00",
    end_hm: str = "17:00",
    slot_minutes: int = 10,
    rng_seed: Optional[int] = None,
) -> int:
    """
    Seed random BOOKED appointments for the given YYYY-MM-DD dates.
    - Avoids conflicts (per doctor + time).
    - Each slot duration: slot_minutes (default 10).
    - Returns count of new appointments created.
    """
    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)

    doctors = _load_doctors()
    appts = _load_all_appointments()
    created = 0

    for ds in dates:
        d = _parse_date(ds)
        slots = _generate_slots_for_date(d, start_hm, end_hm, slot_minutes)
        for doc in doctors:
            # Existing taken slots for this doctor
            taken = {
                str(a.get("time", "")).strip().lower()
                for a in appts
                if _normalize_doctor_name(a.get("doctor", "")) == _normalize_doctor_name(doc["name"])
                and a.get("status") != "CANCELLED"
            }
            free_slots = [s for s in slots if s.strip().lower() not in taken]
            if not free_slots:
                continue
            k = min(per_doctor, len(free_slots))
            chosen = rng.sample(free_slots, k=k)

            for ts in chosen:
                aid = _get_next_appointment_id(appts)
                appt = {
                    "appointment_id": aid,
                    "patient_name": f"Patient-{rng.randint(1000, 9999)}",
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
) -> int:
    """
    Seed random appointments for an inclusive date range [start_date, end_date], YYYY-MM-DD.
    """
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
    )


def seed_random_appointments_days(
    days: int,
    per_doctor: int = 6,
    start_hm: str = "09:00",
    end_hm: str = "17:00",
    slot_minutes: int = 10,
    rng_seed: Optional[int] = None,
) -> int:
    """
    Seed random appointments for `days` days including today.
    - For today, only slots strictly after current time are considered.
    - Avoids conflicts (per doctor + time).
    - Returns count of new appointments created.
    """
    if days <= 0:
        return 0

    _ensure_dataset_seeded()
    rng = random.Random(rng_seed)

    doctors = _load_doctors()
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
            free_slots = [s for s in slots if s.strip().lower() not in taken]
            if not free_slots:
                continue

            k = min(per_doctor, len(free_slots))
            chosen = rng.sample(free_slots, k=k)

            for ts in chosen:
                aid = _get_next_appointment_id(appts)
                appt = {
                    "appointment_id": aid,
                    "patient_name": f"Patient-{rng.randint(1000, 9999)}",
                    "doctor": doc["name"],
                    "department": doc.get("department", ""),
                    "time": ts,
                    "status": "BOOKED",
                }
                appts.append(appt)
                created += 1

    _save_all_appointments(appts)
    return created


if __name__ == "__main__":
    # Simple CLI:
    #   python MakeDataBase.py 2025-01-01 2025-01-02 --per-doctor 4
    # or range:
    #   python MakeDataBase.py 2025-01-01 2025-01-05 --range --per-doctor 5
    import argparse

    parser = argparse.ArgumentParser(description="Seed random appointments (10-minute slots) or clear database.")
    parser.add_argument("--clear-appointments", action="store_true", help="Reset appointments.json to empty list and exit.")
    parser.add_argument("--clear-all", action="store_true", help="Delete both doctors.json and appointments.json, then re-seed doctors.json by default and exit.")
    parser.add_argument("--no-reseed", action="store_true", help="With --clear-all, do not re-seed doctors.json.")
    parser.add_argument("--days", type=int, default=3, help="Number of days including today to seed.")
    parser.add_argument("date", nargs="*", help="Dates YYYY-MM-DD. If --range, provide START END.")
    parser.add_argument("--range", action="store_true", help="Interpret provided dates as inclusive range.")
    parser.add_argument("--per-doctor", type=int, default=6, help="Slots to book per doctor per date.")
    parser.add_argument("--start", default="09:00", help="Day start time HH:MM (default 09:00).")
    parser.add_argument("--end", default="17:00", help="Day end time HH:MM (default 17:00).")
    parser.add_argument("--slot-minutes", type=int, default=10, help="Slot duration (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()
    _ensure_dataset_seeded()
    args.clear_appointments=True

    if args.clear_all:
        stats = clear_all(reseed_doctors=(not args.no_reseed))
        print(f"Cleared database: removed {stats['appointments_removed']} appointments. appointments.json existed={stats['appointments_file']} doctors.json existed={stats['doctors_file']} reseeded_doctors={stats['reseeded']}")

    if args.clear_appointments:
        n = clear_appointments()
        print(f"Cleared appointments.json; removed {n} appointments.")


    if args.days is not None:
        if args.days <= 0:
            print("No appointments created: --days must be >= 1.")
            created = 0
        else:
            created = seed_random_appointments_days(
                args.days,
                per_doctor=args.per_doctor,
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
            start_hm=args.start,
            end_hm=args.end,
            slot_minutes=args.slot_minutes,
            rng_seed=args.seed,
        )
    else:
        if not args.date:
            raise SystemExit("Provide --days N or one/more dates YYYY-MM-DD (or --range START END).")
        created = seed_random_appointments_for_dates(
            args.date,
            per_doctor=args.per_doctor,
            start_hm=args.start,
            end_hm=args.end,
            slot_minutes=args.slot_minutes,
            rng_seed=args.seed,
        )

    print(f"Created {created} random appointments.")