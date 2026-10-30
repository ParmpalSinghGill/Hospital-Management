"""Offline tests for llm_transport._race (hedged requests)."""
import asyncio

import pytest

from llm_transport import _race


def _calls(*specs):
    """Each spec: (delay_s, result_or_exception). Returns make_call + start/cancel log."""
    log = {"started": [], "cancelled": []}
    it = iter(enumerate(specs))

    async def make_call():
        i, (delay, outcome) = next(it)
        log["started"].append(i)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            log["cancelled"].append(i)
            raise
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return make_call, log


def run(coro):
    return asyncio.run(coro)


def test_single_call_passthrough():
    make_call, log = _calls((0, "a"))
    assert run(_race(make_call, 1, 0, "t")) == "a"
    assert log["started"] == [0]


def test_fastest_copy_wins_and_rest_cancelled():
    make_call, log = _calls((0.2, "slow"), (0.01, "fast"), (0.3, "slower"))
    assert run(_race(make_call, 3, 0, "t")) == "fast"
    assert log["started"] == [0, 1, 2]
    assert sorted(log["cancelled"]) == [0, 2]


def test_failed_copy_falls_back_to_other():
    make_call, log = _calls((0.01, RuntimeError("boom")), (0.05, "ok"))
    assert run(_race(make_call, 2, 0, "t")) == "ok"


def test_all_copies_fail_raises():
    make_call, _ = _calls((0.01, RuntimeError("a")), (0.02, ValueError("b")))
    with pytest.raises(ValueError):
        run(_race(make_call, 2, 0, "t"))


def test_stagger_skips_duplicate_when_first_is_fast():
    make_call, log = _calls((0.01, "fast"), (0, "unused"))
    assert run(_race(make_call, 2, 200, "t")) == "fast"
    assert log["started"] == [0]


def test_stagger_launches_duplicate_when_first_is_slow():
    make_call, log = _calls((0.5, "slow"), (0.01, "rescue"))
    assert run(_race(make_call, 2, 50, "t")) == "rescue"
    assert log["started"] == [0, 1]
    assert log["cancelled"] == [0]


def test_stagger_failure_launches_next_immediately():
    make_call, log = _calls((0.01, RuntimeError("x")), (0.01, "ok"))
    assert run(_race(make_call, 2, 5000, "t")) == "ok"
