"""Industry vocabulary layer: profile resolution, settings round-trip, prompts."""
from __future__ import annotations

import pytest

import industry_profile as ip


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("INDUSTRY", raising=False)
    monkeypatch.delenv("BUSINESS_NAME", raising=False)


def test_all_industries_registered():
    assert set(ip.INDUSTRY_IDS) == {"hospital", "restaurant", "hotel"}


@pytest.mark.parametrize(
    "industry,resource,booking,customer",
    [
        ("hospital", "doctor", "appointment", "patient"),
        ("restaurant", "table", "reservation", "guest"),
        ("hotel", "room", "booking", "guest"),
    ],
)
def test_vocabulary_per_industry(industry, resource, booking, customer):
    p = ip.get_profile(industry)
    assert p.resource_label == resource
    assert p.booking_label == booking
    assert p.customer_label == customer


def test_unknown_industry_falls_back_to_hospital():
    assert ip.get_profile("casino").id == "hospital"
    assert ip.get_profile("").id == "hospital"
    assert ip.get_profile(None).id == "hospital"


def test_current_reads_env(monkeypatch):
    monkeypatch.setenv("INDUSTRY", "hotel")
    assert ip.current().id == "hotel"
    # Re-read each call, so an Admin change takes effect without a restart.
    monkeypatch.setenv("INDUSTRY", "restaurant")
    assert ip.current().id == "restaurant"


def test_only_hospital_supports_prescriptions():
    assert ip.get_profile("hospital").supports_prescriptions is True
    assert ip.get_profile("restaurant").supports_prescriptions is False
    assert ip.get_profile("hotel").supports_prescriptions is False


def test_describe_and_all_profiles_are_json_safe():
    import json

    payload = {"active": ip.describe("hotel"), "available": ip.all_profiles()}
    assert json.loads(json.dumps(payload))["active"]["id"] == "hotel"
    assert len(payload["available"]) == 3


def test_every_profile_has_examples_and_categories():
    for pid in ip.INDUSTRY_IDS:
        p = ip.get_profile(pid)
        assert p.examples, f"{pid} needs example prompts for the landing page"
        assert p.categories, f"{pid} needs categories"


class TestSettings:
    def test_industry_round_trip(self, monkeypatch, tmp_path):
        import service_settings as ss

        monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "admin_settings.json")
        saved = ss.save_settings({"industry": "restaurant", "business_name": "Copper Fork"})
        assert saved["industry"] == "restaurant"
        assert saved["business_name"] == "Copper Fork"
        # save_settings pushes to env so running pipelines pick it up.
        import os

        assert os.environ["INDUSTRY"] == "restaurant"
        assert os.environ["BUSINESS_NAME"] == "Copper Fork"

    def test_invalid_industry_rejected(self, monkeypatch, tmp_path):
        import service_settings as ss

        monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "admin_settings.json")
        with pytest.raises(ValueError):
            ss.save_settings({"industry": "casino"})

    def test_blank_business_name_falls_back(self, monkeypatch, tmp_path):
        import service_settings as ss

        monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "admin_settings.json")
        assert ss.save_settings({"business_name": "   "})["business_name"] == "ReceptionAi"

    def test_industry_is_offered_in_options(self):
        import service_settings as ss

        ids = [o["id"] for o in ss.options_catalog()["industry"]]
        assert ids == list(ip.INDUSTRY_IDS)


class TestAgentPrompts:
    """The specialist prompts must speak the configured industry's nouns."""

    @staticmethod
    def _prompts(monkeypatch, industry: str) -> dict[str, str]:
        # Building a ReAct agent is offline; the key only has to exist.
        monkeypatch.setenv("GROQ_API_KEY", "test-dummy-key")
        monkeypatch.setenv("INDUSTRY", industry)
        monkeypatch.setenv("CLI_LLM_PROVIDER", "groq")
        import reception_agents as ha

        ha.AGENT_SYSTEM_PROMPTS.clear()
        for build in (
            ha.build_router_agent,
            ha.build_general_agent,
            ha.build_booking_agent,
            ha.build_cancellation_agent,
            ha.build_reschedule_agent,
            ha.build_prescription_agent,
        ):
            build()
        return dict(ha.AGENT_SYSTEM_PROMPTS)

    @pytest.mark.parametrize(
        "industry,word", [("restaurant", "table"), ("hotel", "room"), ("hospital", "doctor")]
    )
    def test_booking_prompt_uses_industry_noun(self, monkeypatch, industry, word):
        assert word in self._prompts(monkeypatch, industry)["booking"].lower()

    def test_all_specialists_built(self, monkeypatch):
        assert set(self._prompts(monkeypatch, "hotel")) == {
            "router",
            "general",
            "booking",
            "cancelling",
            "rescheduling",
            "prescriptions",
        }

    def test_non_medical_industries_disable_prescriptions(self, monkeypatch):
        for industry in ("restaurant", "hotel"):
            rx = self._prompts(monkeypatch, industry)["prescriptions"]
            assert "get_prescriptions" not in rx
            assert "no medical or prescription records" in rx

    def test_hospital_keeps_prescription_tool(self, monkeypatch):
        assert "get_prescriptions" in self._prompts(monkeypatch, "hospital")["prescriptions"]

    def test_router_omits_prescriptions_for_non_medical(self, monkeypatch):
        assert "'prescriptions'" not in self._prompts(monkeypatch, "restaurant")["router"]
        assert "'prescriptions'" in self._prompts(monkeypatch, "hospital")["router"]

    def test_business_name_appears_in_greeting(self, monkeypatch):
        monkeypatch.setenv("BUSINESS_NAME", "Copper Fork")
        assert "Copper Fork" in self._prompts(monkeypatch, "restaurant")["general"]


class TestRealtimePrompt:
    def test_realtime_prompt_follows_industry(self, monkeypatch):
        from voice_bridge import realtime_system_prompt

        monkeypatch.setenv("INDUSTRY", "hotel")
        assert "room" in realtime_system_prompt().lower()
        monkeypatch.setenv("INDUSTRY", "restaurant")
        assert "table" in realtime_system_prompt().lower()


class TestLanding:
    def test_renders_for_every_industry(self, monkeypatch):
        from landing import render

        for pid in ip.INDUSTRY_IDS:
            monkeypatch.setenv("INDUSTRY", pid)
            html = render()
            assert "ReceptionAi" in html
            # All three verticals are advertised regardless of the active one.
            for other in ip.INDUSTRY_IDS:
                assert ip.get_profile(other).label in html
            # Exactly one card is flagged as the running deployment.
            assert html.count("Running now") == 1
            # No unrendered f-string braces leaked from the CSS.
            assert "{{" not in html and "}}" not in html

    def test_active_industry_examples_shown(self, monkeypatch):
        from landing import render

        monkeypatch.setenv("INDUSTRY", "restaurant")
        html = render()
        assert "patio" in html.lower()

    def test_hero_transcript_uses_active_industry(self, monkeypatch):
        from landing import render

        monkeypatch.setenv("INDUSTRY", "hotel")
        assert "room from Friday" in render()
        monkeypatch.setenv("INDUSTRY", "restaurant")
        assert "table for four" in render()

    def test_call_cta_deep_links_to_autostart(self, monkeypatch):
        """Every 'Request a call' must carry ?call=1 so /app/ dials on arrival."""
        from landing import render

        monkeypatch.setenv("INDUSTRY", "hospital")
        html = render()
        assert html.count('href="/app/?call=1"') >= 3
        assert "Request a call" in html

    def test_sample_content_is_labelled(self, monkeypatch):
        """Demo testimonials must stay visibly marked as sample content."""
        from landing import render

        monkeypatch.setenv("INDUSTRY", "hospital")
        html = render()
        assert "Sample" in html
        assert "replace with your own" in html.lower()
