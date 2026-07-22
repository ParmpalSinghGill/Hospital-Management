"""Facade / import compatibility tests after the module split."""

from __future__ import annotations

import database
from Main import GraphState, build_graph, build_booking_agent
from Tools import book_appointment, list_doctors, lookup_patient
from tool_catalog import ALL_TOOLS, build_tool_catalog


def test_database_facade_exports():
    for name in (
        "init_db",
        "_load_doctors",
        "_load_patients",
        "_load_all_appointments",
        "normalize_appointment_time",
        "department_matches",
        "check_slot_bookable",
        "find_nearest_available_times",
        "_resolve_patient",
        "_find_prescriptions",
    ):
        assert hasattr(database, name), name


def test_tools_facade_and_catalog():
    assert lookup_patient.name == "lookup_patient"
    assert list_doctors.name == "list_doctors"
    assert book_appointment.name == "book_appointment"
    assert len(ALL_TOOLS) == 7
    catalog = build_tool_catalog()
    assert catalog["ok"] is True
    assert catalog["count"] == 7
    names = {t["name"] for t in catalog["tools"]}
    assert names == {
        "lookup_patient",
        "save_patient",
        "list_doctors",
        "book_appointment",
        "cancel_appointment",
        "reschedule_appointment",
        "get_prescriptions",
    }


def test_main_exports_graph_builder():
    assert GraphState is not None
    assert callable(build_booking_agent)
    graph = build_graph()
    assert graph is not None
    # compiled graph should expose invoke
    assert hasattr(graph, "invoke")
