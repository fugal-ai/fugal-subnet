"""The health rules' coldkey minimums follow the serving field
(menu_state.retire_minimum): in a field too small for the compiled
minimums, an entry whose host left the listing, or whose calls fail for
the field's majority, still retires; one operator never retires an entry
alone, however many hotkeys it runs (RETIRE_MIN_FLOOR)."""
from __future__ import annotations

import pytest

from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from tests import menu_fixtures as fx


def test_the_minimum_is_a_majority_of_the_field_between_the_floor_and_the_compiled_one():
    assert ms.RETIRE_MIN_FLOOR == 2
    fail, vote = ms.RETIRE_FAIL_MIN_COLDKEYS, ms.ENDPOINT_MIN_VOTERS
    assert [ms.retire_minimum(n, fail) for n in (0, 1, 2, 3, 4, 5, 6, 7, 50)] == \
        [2, 2, 2, 2, 3, 3, 4, 4, 4]
    assert [ms.retire_minimum(n, vote) for n in (1, 2, 3, 4, 5, 50)] == [2, 2, 2, 3, 3, 3]


def _window(coldkeys: int, *, failing=None, gone=True,
            epochs: int = 12, honest_extra: int = 0):
    """`epochs` summaries of a field of `coldkeys` coldkeys (a coldkey is one
    voter however many hotkeys it runs: menu_inputs.endpoint_vote, and the
    miner attack suite's many-hotkey cases) and `honest_extra` more that
    read the dead entry's host listed and answer it. The dead entry (only
    pinned host host-c) is read unlisted by the first `coldkeys` when
    `gone`, and its calls fail for the coldkeys in `failing` (all of the
    first `coldkeys` by default). Other entries are listed and answered, so
    no epoch is an outage."""
    st = fx.wide_state(epoch=200)
    dead = next(e for e in st.entries if e.id == "lab-b/model-two")
    failing = set(range(coldkeys)) if failing is None else set(failing)
    window = []
    for k in range(200 - epochs, 200):
        votes = tuple(ms.HostRead(model=dead.model, host=h, voters=coldkeys + honest_extra,
                                  listed=not gone or honest_extra > coldkeys,
                                  price_in=0 if gone else 1, price_out=0 if gone else 1,
                                  limit=0 if gone else 1000)
                      for h in dead.provider_order) if gone else ()
        seen = {(dead.model, h) for h in dead.provider_order} if gone else set()
        for e in st.entries:
            for h in e.provider_order:
                if (e.model, h) not in seen:
                    seen.add((e.model, h))
                    votes += (ms.HostRead(model=e.model, host=h, voters=coldkeys + honest_extra,
                                          listed=True, price_in=1, price_out=1, limit=1000),)
        calls = tuple(mi.CallCount(entry_id=e.id, rev=e.rev, coldkey=f"ck{i}", calls=10,
                                   failed=10 if (e is dead and i in failing) else 0)
                      for i in range(coldkeys + honest_extra) for e in st.entries)
        window.append(mi.EpochSummary(epoch=k, votes=votes, calls=calls, pairs=(), use=()))
    return st, dead, window


def _verdict(st, dead, window):
    h = {x.entry_id: x for x in mi.health_stats(st, window, through=199)}[dead.id]
    return h, ms.health_retires(h, fx.PARAMS)


@pytest.mark.parametrize("coldkeys, retires", [(1, None), (2, ms.REASON_FAILING),
                                               (3, ms.REASON_FAILING), (4, ms.REASON_FAILING)])
def test_an_entry_nobody_can_call_retires_in_any_field_of_two_or_more(coldkeys, retires):
    st, dead, window = _window(coldkeys)
    h, why = _verdict(st, dead, window)
    assert h.field == coldkeys and h.unlisted_epochs == (12 if coldkeys >= 2 else 0)
    assert why == retires


@pytest.mark.parametrize("coldkeys", [2, 3])
def test_gone_retires_on_the_fields_votes_when_no_call_is_made(coldkeys):
    """Nobody calls the entry (its calls are not made), the field's every
    coldkey reads its host unlisted: gone after RETIRE_GONE_EPOCHS."""
    st, dead, window = _window(coldkeys, failing=())
    h, why = _verdict(st, dead, window)
    assert h.failing == 0 and h.unlisted_epochs == 12 and why == ms.REASON_GONE


def test_one_operator_alone_never_retires_an_entry():
    st, dead, window = _window(1)
    h, why = _verdict(st, dead, window)
    assert h.field == 1 and why is None


def test_one_dishonest_operator_in_a_field_of_three_fails_its_own_calls_and_retires_nothing():
    """Its reads are attested, so it cannot read a listed host unlisted; it
    can only drop its own calls: one failing coldkey of three."""
    st, dead, window = _window(1, gone=False, honest_extra=2)
    h, why = _verdict(st, dead, window)
    assert (h.field, h.eligible, h.failing, h.unlisted_epochs) == (3, 3, 1, 0) and why is None


def test_a_large_field_keeps_the_compiled_minimums():
    st, dead, window = _window(3, gone=False, honest_extra=7)       # 3 of 10 fail
    h, why = _verdict(st, dead, window)
    assert h.field == 10 and ms.retire_minimum(h.field, ms.RETIRE_FAIL_MIN_COLDKEYS) == 4
    assert why is None
    # Without a field, the step reads the compiled minimum as before.
    assert ms.health_retires(ms.HealthStat(dead.id, 1, eligible=3, failing=3), fx.PARAMS) is None
    assert ms.health_retires(ms.HealthStat(dead.id, 1, eligible=3, failing=3, field=3),
                             fx.PARAMS) == ms.REASON_FAILING


def test_the_step_refuses_a_health_stat_with_more_eligible_than_serving():
    h = ms.HealthStat("lab-b/model-two", 1, eligible=3, failing=3, field=2)
    with pytest.raises(ValueError, match="eligible than serving"):
        ms.step(fx.state(), fx.inputs(health=(h,)), fx.PARAMS)
