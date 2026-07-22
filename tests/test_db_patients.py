"""Unit tests for patient identity / phone helpers."""

from __future__ import annotations

import json

import db_patients


def test_phone_normalize_and_complete():
    assert db_patients._normalize_phone("+91-98765-43210") == "919876543210"
    assert db_patients._phone_is_complete("9876543210")
    assert not db_patients._phone_is_complete("98765")


def test_phones_match_ignores_country_code():
    assert db_patients._phones_match("+919876543210", "9876543210")
    assert db_patients._phones_match("09876543210", "9876543210")
    assert not db_patients._phones_match("9876543210", "9876543211")


def test_names_match():
    assert db_patients._names_match("Even Sharma", "even sharma")
    assert not db_patients._names_match("Even", "Even Sharma")


def test_get_or_create_and_find_by_phone(isolated_db):
    created = db_patients._get_or_create_patient("Ada Lovelace", "9876543210", "Pune")
    assert created["patient_id"].startswith("PAT-")
    found = db_patients._find_patient_by_phone("+91 9876543210")
    assert found is not None
    assert found["patient_id"] == created["patient_id"]
    assert found["name"] == "Ada Lovelace"


def test_resolve_patient_incomplete_phone(isolated_db):
    result = db_patients._resolve_patient(phone="91382292")
    assert result["ok"] is False
    assert result.get("incomplete_phone") is True


def test_resolve_patient_new_by_phone(isolated_db):
    result = db_patients._resolve_patient(phone="9123456780")
    assert result["ok"] is True
    assert result.get("is_new") is True


def test_resolve_patient_returning_requires_name_confirm(isolated_db):
    db_patients._get_or_create_patient("Bob Builder", "9111111111")
    result = db_patients._resolve_patient(phone="9111111111")
    assert result["ok"] is True
    assert result.get("is_returning") is True
    assert result.get("confirm_name_from_db") == "Bob Builder"


def test_resolve_patient_name_mismatch(isolated_db):
    db_patients._get_or_create_patient("Bob Builder", "9222222222")
    result = db_patients._resolve_patient(phone="9222222222", patient_name="Alice")
    assert result["ok"] is False
    assert result.get("name_mismatch") is True


def test_name_alone_is_not_identity(isolated_db):
    db_patients._get_or_create_patient("Unique Name", "9333333333")
    result = db_patients._resolve_patient(patient_name="Unique Name")
    assert result["ok"] is False
    assert result.get("needs_phone") is True
