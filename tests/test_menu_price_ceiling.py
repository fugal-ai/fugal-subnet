"""Protocol v4: the price ceiling of an entry (menu_rules.check_price) and the
realised-cost cap on a provisional entry's exploration calls in the TD
(harness.PROVISIONAL_SPEND_CAP_USD).

The ceiling is applied by the root's admission, by every node's and TD's
`check_delta` (which re-runs the admission), by the menu step's refresh (an
entry whose refreshed price would pass it leaves, reason `price_ceiling`),
and by the TD's own listing read of a provisional entry (`td_v4.overpriced`).
No network, no spend.
"""
from __future__ import annotations

import dataclasses
import json
import threading
from fractions import Fraction

import pytest

from fugal_subnet import chain_proposals as cp
from fugal_subnet import menu_rules as mr
from fugal_subnet import menu_state as ms
from fugal_subnet.tee import endpoint_reads as er
from tests import menu_fixtures as fx

DEAR = "lab-b/model-dear"


def _admit_at(price_in, price_out):
    base = fx.wide_state(epoch=101)
    hk, raw = "hk-1", fx.document(proposer="hk-1")[1]
    p = cp.proposal_hash(raw)
    read = cp.ChainRead(epoch=101, block=101 * 360 - 30, block_hash="ab" * 32,
                        proposals=((hk, p),), coldkey_of={hk: "ck-1"},
                        serving=frozenset({hk}), reveal=("e00000099", "cd" * 32))
    listing = {p: fx.listing(price_in=price_in, price_out=price_out)}
    return base, cp.admit(base, read, {p: raw}, listing, fx.ALLOW, fx.PARAMS)


def test_the_ceiling_has_headroom_over_the_dearest_entry_of_the_pinned_seed():
    seed = json.loads(ms.SEED_PATH.read_text())
    dearest_in = max(e["price_in"] for e in seed["entries"])
    dearest_out = max(e["price_out"] for e in seed["entries"])
    assert (dearest_in, dearest_out) == (8_000_000, 40_000_000)      # $8 / $40 per M
    assert mr.MAX_PRICE_IN_PICO >= 3 * dearest_in
    assert mr.MAX_PRICE_OUT_PICO >= 2.5 * dearest_out
    assert all(mr.check_price(e["price_in"], e["price_out"]) is None for e in seed["entries"])


@pytest.mark.parametrize("price_in, price_out, joins", [
    (mr.MAX_PRICE_IN_PICO, mr.MAX_PRICE_OUT_PICO, True),             # at the ceiling
    (mr.MAX_PRICE_IN_PICO + 1, 400_000, False),
    (100_000, mr.MAX_PRICE_OUT_PICO + 1, False),
    (150_000_000, 600_000_000, False),                               # $150 / $600 per M
    (er.MAX_PRICE_PICO, er.MAX_PRICE_PICO, False),                   # the listing reader's bound
])
def test_the_admission_refuses_an_entry_above_the_ceiling(price_in, price_out, joins):
    base, delta = _admit_at(price_in, price_out)
    verdict = delta.considered[0].verdict
    assert verdict == (cp.JOINED if joins else cp.REFUSED + "price_above_ceiling")
    assert len(cp.check_delta(base, cp.delta_bytes(delta), fx.ALLOW, params=fx.PARAMS)) == joins


def test_every_node_refuses_a_delta_that_admits_an_entry_above_the_ceiling():
    base, delta = _admit_at(150_000_000, 600_000_000)
    forged = dataclasses.replace(delta, considered=tuple(
        dataclasses.replace(c, verdict=cp.JOINED) for c in delta.considered))
    with pytest.raises(cp.DeltaRefused):
        cp.check_delta(base, cp.delta_bytes(forged), fx.ALLOW, params=fx.PARAMS)


def _dear_state(epoch=101, price_out=30_000_000):
    return fx.state(*fx.state(epoch=epoch).entries,
                    fx.entry(DEAR, None, provider_order=("host-c",), price_in=1_000_000,
                             price_out=price_out), epoch=epoch)


def test_a_refresh_that_would_pass_the_ceiling_retires_the_entry_and_the_reader_accepts_it():
    prev = _dear_state()
    reads = (fx.read(DEAR, "host-c", price_in=1_000_000, price_out=120_000_000),)
    out = ms.step_detail(prev, fx.inputs(epoch=100, host_reads=reads), fx.PARAMS)
    assert DEAR not in out.state.ids()
    r = {x.id: x for x in out.state.retired}[DEAR]
    assert r.reason == ms.REASON_PRICE_CEILING
    assert ms.check_transition(prev, out.state, fx.PARAMS) is None
    # Within the ceiling the price moves as before.
    ok = (fx.read(DEAR, "host-c", price_in=1_000_000, price_out=100_000_000),)
    moved = ms.step_detail(prev, fx.inputs(epoch=100, host_reads=ok), fx.PARAMS).state
    assert moved.entry(DEAR).price_out == 100_000_000


def test_a_refresh_above_the_ceiling_never_leaves_fewer_than_the_minimum():
    prev = _dear_state()
    reads = (fx.read(DEAR, "host-c", price_in=1_000_000, price_out=120_000_000),)
    params = dataclasses.replace(fx.PARAMS, min_entries=len(prev.entries))
    out = ms.step_detail(prev, fx.inputs(epoch=100, host_reads=reads), params)
    assert out.state.entry(DEAR).price_out == 30_000_000             # nothing moved
    assert ("retire_waits", DEAR, ms.REASON_PRICE_CEILING) in [
        (e.kind, e.subject, e.detail) for e in out.events]


def test_the_reader_refuses_a_ceiling_retirement_no_refresh_could_give():
    prev = _dear_state(price_out=10_000_000)                         # x4 is $40/M: within
    gone = dataclasses.replace(
        ms.carry(prev), serial=prev.serial + 1,
        entries=tuple(e for e in prev.entries if e.id != DEAR),
        retired=(ms.Retired(DEAR, prev.epoch + 1, ms.REASON_PRICE_CEILING,
                            prev.epoch + 1 + fx.PARAMS.cooldown_epochs),))
    assert "ceiling" in (ms.check_transition(prev, gone, fx.PARAMS) or "")


def test_the_td_does_not_call_a_provisional_entry_its_own_read_prices_above_the_ceiling():
    from fugal_subnet.tee import td_v4

    state = fx.provisional(fx.wide_state(epoch=101), price_in=20_000_000, price_out=90_000_000)
    e = next(x for x in state.entries if x.provisional)

    def read(p_in, p_out):
        return er.EndpointRead(model=e.model, hosts=tuple(
            er.HostListing(host=h, listed=True, price_in=p_in, price_out=p_out, limit=131072)
            for h in e.provider_order))
    assert td_v4.overpriced(state, [read(20_000_000, 90_000_000)]) == frozenset()
    assert td_v4.overpriced(state, [read(20_000_000, 100_000_001)]) == {e.id}


# --- the realised-cost cap on a provisional entry's exploration ------------------------------


class _Record:
    def __init__(self, rid, prompt_tokens, completion_tokens):
        self.request_id, self.prompt_tokens = rid, prompt_tokens
        self.completion_tokens, self.cost_usd = completion_tokens, 0
        self.billed_nano, self.served_by = None, "host"


class _Proxy:
    def __init__(self):
        self.records, self.prices = [], {}

    def clear(self):
        self.records = []


def test_a_provisional_entrys_exploration_stops_at_the_realised_spend_cap(monkeypatch):
    """Every answer runs to the entry's answer limit: the calls stop once
    the next attempt could pass the cap, and what was spent is within it."""
    from fugal_subnet.tee import harness, td_v4

    eid = "lab-c/model-new@high"
    prices = {eid: (mr.MAX_PRICE_IN_PICO, mr.MAX_PRICE_OUT_PICO, 32_768)}
    spend = harness._ProvisionalSpend(prices)
    proxy = _Proxy()
    lock = threading.Lock()

    def call(proxy_, price_key, q, request=None, deadline_s=None):
        with lock:
            proxy_.records.append(_Record(harness.current_request_id(), 400, 32_768))
        return "answer"

    monkeypatch.setattr(harness, "_call_model", call)
    monkeypatch.setattr(td_v4, "check_wire_request", lambda r: None)
    items = [({"question_id": f"q{i}", "prompt": "x" * 300}, eid, "explore", {}, eid)
             for i in range(40)]
    out = harness._call_all_v4(proxy, items, "e00000101", provisional=frozenset({eid}),
                               spend=spend)
    made = sum(not o.failed for o in out)
    per_call = Fraction(mr.MAX_PRICE_IN_PICO * 400 + mr.MAX_PRICE_OUT_PICO * 32_768, 10**12)
    assert spend.spent == made * per_call
    assert spend.spent <= Fraction(harness.PROVISIONAL_SPEND_CAP_USD)
    assert 0 < made < len(items)                         # a ceiling-priced, verbose entry
    assert spend.held == 0
    # The cap admits one attempt of the dearest entry the rules allow.
    worst = harness._ProvisionalSpend(
        {eid: (mr.MAX_PRICE_IN_PICO, mr.MAX_PRICE_OUT_PICO, mr.MAX_ANSWER_TOKENS)}).worst(
            eid, {"prompt": "x" * 2000})
    assert worst <= Fraction(harness.PROVISIONAL_SPEND_CAP_USD)


def test_provisional_prices_take_the_dearer_of_the_menu_and_the_tds_read():
    from fugal_subnet.tee import harness

    state = fx.provisional(fx.wide_state(epoch=101), price_in=300_000, price_out=900_000)
    e = next(x for x in state.entries if x.provisional)
    read = er.EndpointRead(model=e.model, hosts=tuple(
        er.HostListing(host=h, listed=True, price_in=200_000, price_out=2_000_000, limit=131072)
            for h in e.provider_order))
    assert harness.provisional_prices(state, [read]) == {e.id: (300_000, 2_000_000,
                                                                e.max_tokens)}
    assert harness.provisional_prices(state, []) == {e.id: (300_000, 900_000, e.max_tokens)}
