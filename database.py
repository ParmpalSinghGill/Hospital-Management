"""Hospital data layer — SQLite store under dataset/hospital.db.

Public helpers keep the same names Tools.py / admin_routes.py / MakeDataBase.py
already use (_load_*, _save_*, lookup helpers, time normalization).

Implementation is split across:
  db_core.py, db_time.py, db_doctors.py, db_patients.py,
  db_appointments.py, db_prescriptions.py
"""

from __future__ import annotations

from db_core import (  # noqa: F401
    APPOINTMENT_SOON_HOURS,
    APPOINTMENTS_DB,
    BASE_DIR,
    DATASET_DIR,
    DB_PATH,
    DEFAULT_LUNCH_END,
    DEFAULT_LUNCH_START,
    DOCTORS_DB,
    PATIENTS_DB,
    PRESCRIPTIONS_DB,
    _CLINIC_END,
    _CLINIC_START,
    _DEFAULT_SLOT_MINUTES,
    _SCHEMA,
    _appointment_status_is_live,
    _appointment_write_lock,
    _db,
    _extract_id_number,
    _initialized,
    _migrate_appointment_unique_index,
    _migrate_json_into,
    _next_id,
    _read_json,
    _row_to_dict,
    clear_table,
    init_db,
)
from db_time import (  # noqa: F401
    _hm_to_minutes,
    _iter_clinic_slots_on_day,
    _next_clinic_slot_on_or_after,
    _parse_clock_fragment,
    _parse_soft_clock_fragment,
    _resolve_relative_date,
    classify_appointment_timing,
    is_within_clinic_hours,
    normalize_appointment_time,
    parse_appointment_datetime,
    parse_availability_anchor,
    resolve_appointment_day,
)
from db_doctors import (  # noqa: F401
    _DEPARTMENT_ALIASES,
    _department_query_terms,
    _get_doctor_by_id,
    _get_doctor_by_name,
    _load_doctors,
    _load_unavailable_blocks,
    _next_unavailable_block_id,
    _normalize_doctor_name,
    _save_doctors,
    add_doctor_unavailable,
    department_matches,
    ensure_default_lunch_breaks,
    get_doctor_day_grid,
    is_doctor_unavailable,
    remove_doctor_unavailable,
)
from db_patients import (  # noqa: F401
    MIN_PHONE_DIGITS,
    _find_patient_by_name,
    _find_patient_by_phone,
    _find_patients_by_name,
    _get_next_patient_id,
    _get_or_create_patient,
    _get_patient_by_id,
    _load_patients,
    _names_match,
    _normalize_phone,
    _patient_past_doctors,
    _patient_public,
    _phone_is_complete,
    _phone_lookup_candidates,
    _phone_national_digits,
    _phones_match,
    _resolve_patient,
    _save_patients,
)
from db_appointments import (  # noqa: F401
    _active_booked_times_for_doctor,
    _appointment_counts_for_day,
    _appointment_is_upcoming,
    _complete_past_appointments_for_patient,
    _enrich_appointment,
    _find_active_appointment_for_patient,
    _find_conflict,
    _get_next_appointment_id,
    _insert_appointment,
    _load_all_appointments,
    _patient_ids_with_active_appointment,
    _save_all_appointments,
    _update_appointment,
    build_availability_suggestions,
    build_department_booking_guidance,
    check_slot_bookable,
    find_available_doctors_at_time,
    find_nearest_available_times,
    rank_doctors_for_preferred_time,
)
from db_prescriptions import (  # noqa: F401
    _find_prescriptions,
    _get_next_prescription_id,
    _load_prescriptions,
    _save_prescriptions,
)
