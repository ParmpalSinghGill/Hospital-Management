"""Unit tests for doctors, departments, and unavailable blocks."""

from __future__ import annotations

import db_doctors
import pytest


def test_department_aliases():
    assert db_doctors.department_matches("dental", "Dentistry")
    assert db_doctors.department_matches("heart", "Cardiology")
    assert db_doctors.department_matches("stomach", "Gastroenterology")
    assert not db_doctors.department_matches("ent", "Dentistry")  # ent ⊂ dental trap


def test_normalize_doctor_name():
    assert db_doctors._normalize_doctor_name("Dr. Test Cardio") == "test cardio"
    assert db_doctors._normalize_doctor_name("Dr Test Cardio") == "test cardio"


def test_load_and_get_doctor(isolated_db):
    docs = db_doctors._load_doctors()
    assert len(docs) == 3
    by_id = db_doctors._get_doctor_by_id("DOC-0001")
    assert by_id["name"] == "Dr. Test Cardio"
    by_name = db_doctors._get_doctor_by_name("Dr. Test Dental")
    assert by_name["doctor_id"] == "DOC-0002"


def test_default_lunch_unavailable(isolated_db):
    block = db_doctors.is_doctor_unavailable("DOC-0001", "2026-07-22 14:30")
    assert block is not None
    assert block.get("reason") == "lunch"
    assert db_doctors.is_doctor_unavailable("DOC-0001", "2026-07-22 10:00") is None


def test_add_and_remove_unavailable(isolated_db):
    block = db_doctors.add_doctor_unavailable(
        "DOC-0001",
        "11:00",
        "12:00",
        day="2026-12-01",
        reason="meeting",
    )
    assert block["block_id"]
    assert db_doctors.is_doctor_unavailable("DOC-0001", "2026-12-01 11:30")
    assert db_doctors.remove_doctor_unavailable(block["block_id"])
    assert db_doctors.is_doctor_unavailable("DOC-0001", "2026-12-01 11:30") is None


def test_day_grid_shape(isolated_db):
    grid = db_doctors.get_doctor_day_grid("DOC-0001", "2026-12-02")
    assert grid["day"] == "2026-12-02"
    assert len(grid["hours"]) == 8
    assert len(grid["minutes"]) == 6
    assert grid["stats"]["total"] == 48
    assert grid["stats"]["unavailable"] >= 6  # lunch


def test_day_grid_unknown_doctor(isolated_db):
    with pytest.raises(ValueError):
        db_doctors.get_doctor_day_grid("DOC-9999", "2026-12-02")
