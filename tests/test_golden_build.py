"""Golden dataset builder: state reconstruction and tagging (eval/build_golden.py)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_golden", Path(__file__).resolve().parent.parent / "eval" / "build_golden.py"
)
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)


def _turn(n, user, bot, agent, sent):
    return {
        "turn_number": n,
        "agent": agent,
        "payload": {"user_input_text": user, "bot_response_text": bot},
        "timestamps": {"user_sent_time": sent},
    }


SESSION = {
    "session_id": "sess_x",
    "pipeline_mode": "cascade",
    "interactions": [
        _turn(1, "I have a toothache", "Sure, what's your phone number?", "booking", "2026-07-22T10:00:00.000Z"),
        _turn(2, "9876543210", "Are you Asha Rao?", "booking", "2026-07-22T10:00:10.000Z"),
        _turn(3, "Yes", "What time works for you?", "booking", "2026-07-22T10:00:20.000Z"),
    ],
}
TOOLS = [{
    "at": "2026-07-22T10:00:11.000Z", "tool": "lookup_patient", "session_id": "sess_x",
    "tool_call_id": "call_1", "arguments": {"phone": "9876543210"}, "result": {"ok": True}, "ok": True,
}]


def test_history_holds_prior_turns_and_tool_exchange():
    snaps = bg.snapshots_for_session(SESSION, TOOLS)
    assert [s["user_text"] for s in snaps] == ["I have a toothache", "9876543210", "Yes"]
    assert snaps[0]["history"] == []
    third = snaps[2]["history"]
    roles = [m["role"] for m in third]
    assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"]
    assert third[3]["tool_calls"][0]["name"] == "lookup_patient"
    assert third[4]["tool_call_id"] == third[3]["tool_calls"][0]["id"]


def test_expected_action_is_first_tool_of_the_turn():
    snaps = bg.snapshots_for_session(SESSION, TOOLS)
    assert snaps[1]["expected"]["action"] == "lookup_patient"
    assert snaps[1]["expected"]["llm_calls"] >= 2
    assert snaps[2]["expected"]["action"] == "reply"


def test_sticky_and_tool_tags():
    snaps = bg.snapshots_for_session(SESSION, TOOLS)
    assert {"sticky_short", "tool_turn"} <= set(snaps[1]["categories"])
    assert "sticky_short" in snaps[2]["categories"]


def test_slot_edge_times():
    def cats(text):
        return bg.tag_turn(user_text=text, bot_text="", prev_agent="", agent="booking",
                           rule_route="booking", tools=[], turn_index=0)

    assert "slot_edge" in cats("Can I come at 2:30?")[0]          # lunch 14:30
    assert "slot_edge" in cats("6 pm please")[0]                  # after closing
    assert "slot_edge" in cats("10:45 works")[0]                  # not a 10-minute slot
    assert "slot_edge" not in cats("10:40 works")[0]
    assert "slot_edge" not in cats("my number is 9876543210")[0]  # digits are not a time


def test_other_industry_sessions_are_skipped():
    hotel = {"interactions": [_turn(1, "I need a room for two nights", "Sure", "booking", None)]}
    assert bg._other_industry(hotel)
    assert not bg._other_industry(SESSION)


def test_slot_variants_reuse_a_real_time_question():
    snaps = bg.snapshots_for_session(SESSION, TOOLS)
    extra = bg.slot_variants(snaps + [{**snaps[2], "id": "y#t4", "history": snaps[2]["history"] + [
        {"role": "user", "content": "Yes"}, {"role": "assistant", "content": "What time works for you?"}],
        "source": {**snaps[2]["source"], "session_id": "sess_x"}}], 2, seed=1)
    assert len(extra) == 2
    assert all(v["category"] == "slot_edge" and v["variant_of"] for v in extra)
