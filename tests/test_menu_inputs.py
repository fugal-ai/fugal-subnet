"""Protocol v4: the menu step's inputs from verified proofs (fugal_subnet/menu_inputs.py;
docs/decisions/LIVE_MENU_V4.md, 4.4, 5, 7, 8, 9.2).

Every input is a pure function of the epoch's verified proofs (each with
the coldkey the chain names for its hotkey) and, where the design reads a
window, of earlier epochs' summaries. Votes and counts are per coldkey; the
health rule is the replay's; everything is exact and order-free.
"""
from __future__ import annotations

import dataclasses
import json
import random
from fractions import Fraction

import pytest

from fugal_subnet import explore_pool as xp
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from fugal_subnet.head_odds import ODDS_SCALE
from fugal_subnet.tee.endpoint_reads import EndpointRead, HostListing
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from tests import menu_fixtures as mf

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CATS = {f"q{i:03d}": ("code", "hard", "knowledge", "math")[i % 4] for i in range(400)}


def _r(qid, entry, *, correct=True, failed=False, explore=False, p_tok=100, c_tok=200):
    return QuestionResult(qid, entry, correct and not failed, 0.0,
                          EMPTY if failed else "ab" * 32, p_tok, c_tok,
                          is_exploration=explore, failed=failed)


def _proof(hotkey, results=(), *, reads=(), weights_hash="1f" * 32):
    return BenchmarkProof(
        epoch_id="e00000101", nonce="00" * 32, questions_hash="", weights_hash=weights_hash,
        source_hash="", results=list(results), total_cost_usd=0.0, per_model_costs={},
        attestation_quote=b"", timestamp=0.0, hotkey=hotkey, routing_protocol="x",
        menu_hash="cd" * 32, endpoint_reads=tuple(reads))


def _read(model, *hosts):
    """hosts: (host, listed, price_in, price_out, limit)."""
    return EndpointRead(model, tuple(
        HostListing(h, True, pi, po, lim, "fp8", 0) if listed else HostListing(h, False)
        for h, listed, pi, po, lim in sorted(hosts)))


def _vp(hotkey, coldkey, results=(), **kw):
    return mi.VerifiedProof(_proof(hotkey, results, **kw), coldkey)


# --- the endpoint vote (section 7.1) ------------------------------------------------------


def _price_proofs(attacker_price, attacker_hotkeys=5):
    honest = [_vp(f"5Fh{c}", f"ck-{c}", reads=[_read("lab-a/model-one",
                                                      ("host-a/fp8", True, 100_000, 400_000, 131072))])
              for c in "BC"]
    attacker = [_vp(f"5Fa{i}", "ck-A", reads=[_read(
        "lab-a/model-one", ("host-a/fp8", True, attacker_price, attacker_price, 1))])
        for i in range(attacker_hotkeys)]
    return honest + attacker


def test_one_operators_many_hotkeys_are_one_vote_on_a_price(monkeypatch):
    """A4: one coldkey with five hotkeys against two honest coldkeys. The
    median across coldkeys is the honest price; voting per hotkey would have
    let the one operator set it."""
    (r,) = mi.endpoint_vote(_price_proofs(9_000_000))
    assert (r.voters, r.listed, r.price_in, r.price_out, r.limit) == (3, True, 100_000, 400_000,
                                                                       131072)
    monkeypatch.setattr(mi, "_voter", lambda vp: vp.hotkey)
    (r,) = mi.endpoint_vote(_price_proofs(9_000_000))
    assert r.price_in == 9_000_000 and r.voters == 7


def test_the_median_of_an_even_count_is_rounded_half_up():
    proofs = [_vp(f"5F{i}", f"ck{i}", reads=[_read("lab-b/model-two", ("host-c", True, p, p + 1,
                                                                         lim))])
              for i, (p, lim) in enumerate([(100, 10), (101, 11), (200, 20), (301, 31)])]
    (r,) = mi.endpoint_vote(proofs)
    assert (r.price_in, r.price_out, r.limit, r.voters) == (151, 152, 16, 4)


def test_a_host_is_listed_when_at_least_half_the_coldkeys_read_it_listed():
    def run(n_listed, n_not):
        proofs = [_vp(f"5Fl{i}", f"ckl{i}", reads=[_read("lab-b/model-two",
                                                          ("host-c", True, 100 + i, 200, 50))])
                  for i in range(n_listed)]
        proofs += [_vp(f"5Fn{i}", f"ckn{i}", reads=[_read("lab-b/model-two",
                                                           ("host-c", False, 0, 0, 0))])
                   for i in range(n_not)]
        return mi.endpoint_vote(proofs)[0]

    r = run(3, 1)
    assert r.listed and r.voters == 3 and r.price_in == 101        # unlisted reads carry no price
    r = run(2, 2)
    assert r.listed and r.voters == 2                               # a tie is listed; too few to price
    r = run(1, 3)
    assert not r.listed and r.voters == 4 and (r.price_in, r.limit) == (0, 0)


def test_a_failed_read_is_no_vote():
    proofs = _price_proofs(100_000, attacker_hotkeys=1) + [_vp("5Fnone", "ck-D", reads=[])]
    (r,) = mi.endpoint_vote(proofs)
    assert r.voters == 3


FORGED = {
    "a host listed twice": EndpointRead("lab-a/model-one", (
        HostListing("host-a/fp8", True, 1, 1, 1, "fp8", 0),
        HostListing("host-a/fp8", True, 1, 1, 1, "fp8", 0))),
    "a price that is not whole picodollars": EndpointRead("lab-a/model-one", (
        HostListing("host-a/fp8", True, 1.5, 1, 1, "fp8", 0),)),
    "a price as text": EndpointRead("lab-a/model-one", (
        HostListing("host-a/fp8", True, "0.000001", 1, 1, "fp8", 0),)),
    "a boolean limit": EndpointRead("lab-a/model-one", (
        HostListing("host-a/fp8", True, 1, 1, True, "fp8", 0),)),
    "an unlisted host with a price": EndpointRead("lab-a/model-one", (
        HostListing("host-a/fp8", False, 7, 7, 0, "", 0),)),
}


@pytest.mark.parametrize("name", sorted(FORGED))
def test_a_forged_read_is_no_vote(name, monkeypatch):
    """The proof parser refuses these (endpoint_reads.reads_from_json); the
    vote refuses them again, so a read built any other way moves nothing."""
    honest = _price_proofs(100_000, attacker_hotkeys=0)
    forged = [_vp(f"5Ff{i}", f"ck-f{i}", reads=[FORGED[name]]) for i in range(3)]
    (r,) = mi.endpoint_vote(honest + forged)
    assert (r.voters, r.price_in) == (2, 100_000)
    monkeypatch.setattr(mi, "_read_ok", lambda read: True)
    try:
        got = mi.endpoint_vote(honest + forged)
    except TypeError:
        return                                   # without the rule the vote does not even run
    assert got != (r,)


def test_a_proof_reading_one_model_twice_has_no_vote():
    two = [_read("lab-b/model-two", ("host-c", True, 1, 1, 1)),
           _read("lab-b/model-two", ("host-c", True, 9, 9, 9))]
    assert mi.endpoint_vote([_vp("5F1", "ck1", reads=two)]) == ()


def test_the_vote_is_the_same_in_any_order():
    proofs = _price_proofs(9_000_000) + [
        _vp("5Fx", "ck-X", reads=[_read("lab-b/model-two", ("host-c", True, 5, 6, 7)),
                                  _read("lab-a/model-one", ("host-a/fp8", False, 0, 0, 0),
                                        ("host-b/fp8", True, 3, 4, 5))])]
    want = mi.endpoint_vote(proofs)
    rnd = random.Random(7)
    for _ in range(10):
        rnd.shuffle(proofs)
        assert mi.endpoint_vote(proofs) == want
    assert [(r.model, r.host) for r in want] == sorted((r.model, r.host) for r in want)


def test_two_proofs_of_one_hotkey_are_refused():
    with pytest.raises(ValueError, match="one hotkey"):
        mi.endpoint_vote([_vp("5F1", "ck1"), _vp("5F1", "ck2")])


# --- one epoch's summary ----------------------------------------------------------------


STATE = mf.state()                         # the menu of epoch 101


def _pair(qid, model, y=ODDS_SCALE, p_tok=100, c_tok=200):
    return xp.Pair(qid, model, y, 1, p_tok, c_tok)


def _summary(epoch, *, calls=(), pairs=(), votes=(), use=()):
    return mi.EpochSummary(epoch, tuple(votes), tuple(calls), tuple(pairs), tuple(use))


def test_exploration_calls_are_per_coldkey_and_revision():
    state, p = _with_provisional()
    proofs = [_vp("5F1", "ck1", [_r("q001", HIGH, explore=True),
                                 _r("q002", HIGH, explore=True, failed=True),
                                 _r("q003", LOW, explore=True),
                                 _r("q004", p.id, explore=True, failed=True),
                                 _r("q005", TWO)]),
              _vp("5F2", "ck1", [_r("q006", HIGH, explore=True, failed=True)])]
    got = mi.exploration_calls(proofs, state)
    assert got == (mi.CallCount(HIGH, 1, "ck1", 3, 2), mi.CallCount(LOW, 1, "ck1", 1, 0),
                   mi.CallCount(p.id, 1, "ck1", 1, 1))


def _with_provisional():
    state = mf.provisional(mf.state())
    return state, state.entry("lab-c/model-new@high")


def test_a_provisional_entrys_calls_are_kept_for_its_gates_and_count_against_nothing_else():
    """The outage rule leaves them out: one epoch whose only failures are a
    new entry's calls is not an outage (FAST_CROWN_V4.md 18(d))."""
    state, p = _with_provisional()
    calls = tuple(mi.CallCount(p.id, 1, f"ck{i}", 10, 10) for i in range(4)) + tuple(
        mi.CallCount(HIGH, 1, f"ck{i}", 2, 0) for i in range(4))
    plain = mi.EpochSummary(101, (), tuple(sorted(calls, key=lambda c: (c.entry_id, c.rev,
                                                                         c.coldkey))), (), ())
    assert mi.outage(plain)
    marked = dataclasses.replace(plain, provisional=(p.id,))
    assert not mi.outage(marked)
    assert mi.summary_from_dict(mi.summary_to_dict(marked)) == marked


def test_the_gates_read_the_field_labels_and_calls_from_the_join_on():
    state, p = _with_provisional()
    cells = (mi.PairCell(p.id, 1, "code", False, 30, 20 * ODDS_SCALE, 3000, 9000),
             mi.PairCell(p.id, 1, "math", False, 50, 45 * ODDS_SCALE, 5000, 15000))
    calls = (mi.CallCount(p.id, 1, "ck1", 40, 2), mi.CallCount(p.id, 1, "ck2", 40, 0))
    window = [mi.EpochSummary(100, (), calls, cells, ()),          # before it joined: not read
              mi.EpochSummary(101, (), calls, cells, (), provisional=(p.id,))]
    (g,) = mi.gate_stats(state, window, through=101, mix=mf.MIX)
    assert g.entry_id == p.id and g.rev == 1
    assert g.coldkeys == (ms.CallTally("ck1", 40, 2), ms.CallTally("ck2", 40, 0))
    assert {c.category: (c.answered, c.correct) for c in g.cells} == {"code": (30, 20),
                                                                      "math": (50, 45)}
    assert 0 <= g.lower_ppm <= g.upper_ppm <= ms.PPM
    assert mi.gate_stats(state, window, through=100, mix=mf.MIX) == ()   # joined after it


def test_use_counts_each_counted_head_once_whatever_hotkeys_run_it():
    a = [_r("q001", HIGH), _r("q002", HIGH), _r("q003", TWO), _r("q004", LOW, explore=True)]
    proofs = [_vp("5F1", "ck1", a, weights_hash="aa" * 32),
              _vp("5F2", "ck2", a, weights_hash="aa" * 32),
              _vp("5F3", "ck3", [_r("q001", LOW)], weights_hash="bb" * 32)]
    assert mi.epoch_use(proofs, {"aa" * 32}) == (ms.EntryUse(HIGH, 2), ms.EntryUse(TWO, 1))
    assert mi.epoch_use(proofs, set()) == ()


def test_a_summary_has_one_json_form():
    state, p = _with_provisional()
    proofs = _price_proofs(100_000, attacker_hotkeys=2) + [
        _vp(mf.HOTKEY, "ck9", [_r("q001", HIGH, explore=True, failed=True),
                               _r("q010", p.id, explore=True)])]
    s = mi.summarize_epoch(101, state, proofs, pooled=[_pair("q000", HIGH)], category_of=CATS,
                           counted_heads={"1f" * 32})
    d = mi.summary_to_dict(s)
    text = json.dumps(d, sort_keys=True)
    assert mi.summary_from_dict(json.loads(text)) == s
    for bad in (dict(d, epoch=-1), dict(d, extra=1),
                dict(d, calls=list(reversed(d["calls"]))) if len(d["calls"]) > 1 else dict(d, use=5),
                dict(d, votes=[dict(d["votes"][0], listed=1)]),
                dict(d, pairs=[dict(d["pairs"][0], pairs=1.0)])):
        with pytest.raises(ValueError):
            mi.summary_from_dict(bad)


def test_a_summary_is_typed_field_by_field():
    """Every field has one type: a string where an id, host, model, coldkey
    or category belongs, an integer (never a string of digits) where a count
    belongs, rev at least 1. A summary that is not well-typed is refused
    whole, so the epoch is simply absent from a rebuilt window."""
    good = {"epoch": 100,
            "votes": [{"model": "lab-a/model-one", "host": "host-a/fp8", "voters": 9,
                       "listed": True, "price_in": 1, "price_out": 1, "limit": 1}],
            "calls": [{"entry_id": HIGH, "rev": 1, "coldkey": "ck", "calls": 5, "failed": 1}],
            "pairs": [{"entry_id": HIGH, "rev": 1, "category": "code", "measured": False,
                       "pairs": 3, "y": 1, "tokens_in": 1, "tokens_out": 1}],
            "use": [{"entry_id": HIGH, "picks": 7}],
            "graded": [{"entry_id": HIGH, "rev": 1, "category": "code", "cells": 2,
                        "tokens_in": 1, "tokens_out": 1}]}
    mi.summary_from_dict(good)
    swaps = []
    for table, row in (("votes", good["votes"][0]), ("calls", good["calls"][0]),
                       ("pairs", good["pairs"][0]), ("use", good["use"][0]),
                       ("graded", good["graded"][0])):
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            for bad in ("7", 1.5, True, None) if isinstance(value, int) else (
                    7, ["x"], {"a": 1}, "", "a b", "x" * 300):
                swaps.append((table, key, bad))
    swaps += [("calls", "rev", 0), ("pairs", "rev", 0), ("votes", "voters", -1)]
    for table, key, bad in swaps:
        doc = json.loads(json.dumps(good))
        doc[table][0][key] = bad
        with pytest.raises(ValueError):
            mi.summary_from_dict(doc)
    # The rows the step aggregates never meet a string where it adds.
    state = mf.state(epoch=101)
    doc = json.loads(json.dumps(good))
    doc["calls"][0]["calls"] = "5"
    with pytest.raises(ValueError):
        mi.health_stats(state, [mi.summary_from_dict(doc)], through=100)


def test_a_summary_is_for_the_epoch_of_its_menu():
    with pytest.raises(ValueError, match="menu of epoch"):
        mi.summarize_epoch(102, STATE, [], pooled=[], category_of=CATS, counted_heads=())


# --- token sums and exploration counts (sections 7.2, 8) -----------------------------------


def _cell(entry, cat, n, *, rev=1, measured=False, y=None, t_in=100, t_out=200):
    return mi.PairCell(entry, rev, cat, measured, n, n * ODDS_SCALE if y is None else y,
                       n * t_in, n * t_out)


def test_token_sums_are_of_the_current_revision_and_include_measurement():
    window = [_summary(99, pairs=[_cell(HIGH, "code", 10), _cell(HIGH, "code", 7, rev=2)]),
              _summary(100, pairs=[_cell(HIGH, "math", 5, measured=True, t_in=300),
                                   _cell(TWO, "hard", 4)])]
    got = {s.entry_id: s for s in mi.token_stats(STATE, window, through=100)}
    assert got[HIGH] == ms.TokenStat(HIGH, 1, (ms.TokenCell("code", 10, 1000, 2000),
                                               ms.TokenCell("math", 5, 1500, 1000)))
    assert got[TWO].cells == (ms.TokenCell("hard", 4, 400, 800),)
    assert LOW not in got


def test_exploration_counts_never_include_measurement_and_follow_the_revision():
    window = [_summary(99, pairs=[_cell(HIGH, "code", 10), _cell(HIGH, "code", 7, rev=2),
                                  _cell(HIGH, "math", 300, measured=True)]),
              _summary(100, pairs=[_cell(HIGH, "hard", 5), _cell(TWO, "hard", 4)]),
              _summary(101, pairs=[_cell(HIGH, "hard", 50)])]             # after the scored epoch
    got = mi.explore_counts(STATE, window, through=100)
    assert got == (ms.ExploreCount(HIGH, 1, 15), ms.ExploreCount(LOW, 1, 0),
                   ms.ExploreCount(TWO, 1, 4))
    updated = dataclasses.replace(STATE, entries=tuple(
        dataclasses.replace(e, rev=2) if e.id == HIGH else e for e in STATE.entries))
    assert mi.explore_counts(updated, window, through=100)[0] == ms.ExploreCount(HIGH, 2, 7)


def test_pairs_from_before_an_entry_joined_are_another_run_of_the_id():
    joined = dataclasses.replace(STATE, entries=tuple(
        dataclasses.replace(e, joined_epoch=100, scored_from=124) if e.id == TWO else e
        for e in STATE.entries))
    window = [_summary(99, pairs=[_cell(TWO, "hard", 40)]),
              _summary(100, pairs=[_cell(TWO, "hard", 3)])]
    counts = {c.entry_id: c.pairs for c in mi.explore_counts(joined, window, through=100)}
    assert counts[TWO] == 3


# --- accuracy and bounds (section 5) ---------------------------------------------------------


def _wilson(p, n, z=1.645):
    import math
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


@pytest.mark.parametrize("k, n", [(50, 100), (0, 20), (20, 20), (93, 100), (1, 3)])
def test_one_category_bounds_are_wilsons(k, n):
    rate, lo, hi = mi.weighted_bounds({"code": (n, Fraction(k))}, (("code", 1),))
    want_lo, want_hi = _wilson(k / n, n)
    assert rate == Fraction(k, n)
    assert abs(float(lo) - max(0.0, want_lo)) < 1e-9 and abs(float(hi) - min(1.0, want_hi)) < 1e-9


def test_the_rate_is_weighted_to_the_mix_over_the_categories_measured():
    rate, lo, hi = mi.weighted_bounds({"code": (100, Fraction(90)), "hard": (50, Fraction(25)),
                                       "other": (10, Fraction(0))},
                                      (("code", 15), ("hard", 20), ("math", 45)))
    assert rate == (15 * Fraction(9, 10) + 20 * Fraction(1, 2)) / 35
    assert lo < rate < hi
    assert mi.weighted_bounds({}, (("code", 1),)) == (0, 0, 1)


def test_accuracy_stats_in_ppm():
    window = [_summary(100, pairs=[_cell(HIGH, "code", 100, y=93 * ODDS_SCALE),
                                   _cell(HIGH, "code", 7, rev=2, y=0)])]
    (a,) = mi.accuracy_stats(STATE, window, through=100, mix=mf.MIX)
    lo, _hi = _wilson(0.93, 100)
    assert (a.entry_id, a.rev, a.pairs, a.rate_ppm) == (HIGH, 1, 100, 930_000)
    assert a.lower_ppm == int(lo * 1_000_000)
# --- health (section 9.2, the replay's rule) -------------------------------------------------


def _calls(epoch, per_ck, entry=TWO, extra_ok=0):
    """per_ck: {coldkey: (calls, failed)} to `entry`; every coldkey also
    makes `extra_ok` good calls to HIGH (its total for the outage guard)."""
    calls = [mi.CallCount(entry, 1, ck, n, f) for ck, (n, f) in sorted(per_ck.items())]
    if extra_ok:
        calls += [mi.CallCount(HIGH, 1, ck, extra_ok, 0) for ck in sorted(per_ck)]
    return _summary(epoch, calls=sorted(calls, key=lambda c: (c.entry_id, c.rev, c.coldkey)))


def _health(window, through=111, state=STATE):
    return {h.entry_id: h for h in mi.health_stats(state, window, through=through)}[TWO]


HEALTHY = {f"ck{i}": (1, 0) for i in range(5)}


def test_a_broken_entry_retires_on_a_majority_of_eligible_coldkeys():
    window = [_calls(100 + i, {f"ck{c}": (1, 1) for c in range(5)}, extra_ok=20)
              for i in range(12)]
    h = _health(window)
    assert (h.eligible, h.failing) == (5, 5)
    assert ms.health_retires(h, mf.PARAMS) == ms.REASON_FAILING


def test_one_operator_failing_every_call_retires_nothing(monkeypatch):
    """A5: one coldkey (with five hotkeys' worth of calls) fails every call
    to the entry in every epoch; four others are healthy."""
    def window():
        out = []
        for i in range(12):
            per = {f"ck{c}": (1, 0) for c in range(4)}
            per["ck-attacker"] = (5, 5)
            out.append(_calls(100 + i, per, extra_ok=20))
        return out

    h = _health(window())
    assert (h.eligible, h.failing) == (5, 1)
    assert ms.health_retires(h, mf.PARAMS) is None
    # Counted per hotkey instead of per coldkey, the one operator's five
    # hotkeys would be five failing voters against four healthy ones.
    split = []
    for s in window():
        calls = [c for c in s.calls if c.coldkey != "ck-attacker"]
        calls += [mi.CallCount(e, 1, f"ck-attacker-hk{j}", n, f) for j in range(5)
                  for e, n, f in ((TWO, 1, 1), (HIGH, 20, 0))]
        split.append(dataclasses.replace(s, calls=tuple(sorted(
            calls, key=lambda c: (c.entry_id, c.rev, c.coldkey)))))
    h = _health(split)
    assert ms.health_retires(h, mf.PARAMS) == ms.REASON_FAILING


def test_a_coldkey_with_too_few_calls_is_not_eligible():
    window = [_calls(100 + i, {f"ck{c}": (1, 1) if i < 4 else (0, 0) for c in range(5)})
              for i in range(12)]
    h = _health(window)
    assert (h.eligible, h.failing) == (0, 0)


def test_failures_in_fewer_than_three_epochs_are_not_failing():
    window = [_calls(100 + i, {f"ck{c}": (3, 3) if i in (3, 4) else (1, 0) for c in range(5)},
                     extra_ok=60) for i in range(12)]
    h = _health(window)
    assert h.failing == 0 and h.eligible == 5                     # 6 of 16 failed, in two epochs
    window[5] = _calls(105, {f"ck{c}": (3, 3) for c in range(5)}, extra_ok=60)
    assert _health(window).failing == 5                           # 9 of 18, in three epochs


@pytest.mark.parametrize("bad_epochs", [1, 2, 3])
def test_epochs_in_which_every_call_fails_retire_nothing(bad_epochs):
    """The outage guard: an epoch in which the median coldkey failed more
    than 30% of all its calls is not counted, so one to three such epochs
    retire no healthy entry."""
    window = []
    for i in range(12):
        bad = i >= 12 - bad_epochs
        per = {f"ck{c}": (2, 2 if bad else 0) for c in range(5)}
        s = _calls(100 + i, per, extra_ok=10)
        if bad:
            s = dataclasses.replace(s, calls=tuple(dataclasses.replace(c, failed=c.calls)
                                                   for c in s.calls))
        window.append(s)
    assert sum(mi.outage(s) for s in window) == bad_epochs
    h = _health(window)
    assert h.failing == 0 and ms.health_retires(h, mf.PARAMS) is None


def test_the_window_is_the_last_twelve_counted_epochs():
    window = [_calls(100 + i, {f"ck{c}": (1, 1) for c in range(5)}, extra_ok=20)
              for i in range(3)]                                  # failing, long ago
    window += [_calls(103 + i, HEALTHY, extra_ok=20) for i in range(12)]
    assert _health(window, through=114).failing == 0
    outages = [dataclasses.replace(_calls(115 + i, HEALTHY, extra_ok=20), calls=tuple(
        mi.CallCount(HIGH, 1, f"ck{c}", 20, 20) for c in range(5))) for i in range(5)]
    counted = mi.counted_epochs(window + outages, through=119)
    assert [s.epoch for s in counted] == list(range(103, 115))


def test_health_reads_only_the_entrys_current_revision():
    window = [_calls(100 + i, {f"ck{c}": (1, 1) for c in range(5)}) for i in range(12)]
    window = [dataclasses.replace(s, calls=tuple(dataclasses.replace(c, rev=2) for c in s.calls))
              for s in window]
    assert _health(window).eligible == 0


def _votes(epoch, listed, voters=3, field=6):
    """The vote on lab-b/model-two's one host, beside a listing that shows
    the other entries' hosts (an epoch with most hosts unlisted is a listing
    outage, which "gone" does not count: test_menu_liveness.py), in a field
    of `field` serving coldkeys (their calls to lab-a's entry, all answered:
    the voter minimum follows the field, menu_state.retire_minimum)."""
    others = [ms.HostRead(m, h, 3, True, 1, 1, 1)
              for m in ("lab-a/model-one", "lab-d/model-x") for h in ("host-a/fp8", "host-b/fp8")]
    calls = [mi.CallCount(HIGH, 1, f"ck{c}", 1, 0) for c in range(field)]
    return _summary(epoch, calls=calls,
                    votes=others + [ms.HostRead("lab-b/model-two", "host-c", voters, listed,
                                                *((1, 1, 1) if listed else (0, 0, 0)))])


def test_unlisted_epochs_run_back_from_the_scored_epoch():
    window = [_votes(100, True)] + [_votes(101 + i, False) for i in range(6)]
    assert _health(window, through=106).unlisted_epochs == 6
    window[3] = _votes(103, False, voters=2)                        # too few voters breaks the run
    assert _health(window, through=106).unlisted_epochs == 3
    # In a field of three serving coldkeys two voters are a majority: no break.
    small = [_votes(e.epoch, True, field=3) if e.epoch == 100 else _votes(e.epoch, False, field=3)
             for e in window]
    small[3] = _votes(103, False, voters=2, field=3)
    assert _health(small, through=106).unlisted_epochs == 6
    del window[5]                                                   # a missing epoch breaks it
    assert _health(window, through=106).unlisted_epochs == 1


# --- measurement reports and filings (section 5) ----------------------------------------------
# --- all of them -------------------------------------------------------------------------------


def _epoch_world():
    """Epoch 100's proofs on the menu of epoch 100 (an entry provisional in
    it, joined at 100), a window through it, and the step from the menu of
    epoch 101."""
    epoch_state = mf.provisional(mf.state(epoch=100))
    p = epoch_state.entry("lab-c/model-new@high")
    start = ms.step(epoch_state, mf.inputs(epoch=99), mf.PARAMS)
    reads = [_read("lab-a/model-one", ("host-a/fp8", True, 120_000, 410_000, 100_000),
                   ("host-b/fp8", True, 110_000, 420_000, 131072)),
             _read("lab-b/model-two", ("host-c", True, 2_000_000, 10_000_000, 65536))]
    proofs = []
    for i in range(5):
        res = [_r(f"q{i * 10 + j:03d}", (HIGH, LOW, TWO, p.id)[j % 4], explore=True,
                  failed=(i == 4)) for j in range(12)]
        res += [_r("q300", HIGH)]
        proofs.append(_vp(f"5Fhk{i}", f"ck{i}", res, reads=reads))
    pooled = [_pair(f"q{j:03d}", (HIGH, LOW, TWO, p.id)[j % 4]) for j in range(48)]
    summary = mi.summarize_epoch(100, epoch_state, proofs, pooled=pooled, category_of=CATS,
                                 counted_heads={"1f" * 32})
    earlier = [mi.summary_from_dict(dict(mi.summary_to_dict(summary), epoch=e))
               for e in range(90, 100)]
    return start, epoch_state, earlier + [summary], proofs


def test_the_step_runs_on_inputs_built_from_verified_proofs():
    start, epoch_state, window, proofs = _epoch_world()
    inputs = mi.step_inputs(start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE,
                            lab_list=mf.ALLOW, params=mf.PARAMS, category_of=CATS)
    assert inputs.host_reads == window[-1].votes
    assert {c.entry_id: c.pairs for c in inputs.explore_counts}[HIGH] == 132
    (g,) = inputs.gates
    assert g.entry_id == "lab-c/model-new@high" and len(g.coldkeys) == 5
    out = ms.step_detail(start, inputs, mf.PARAMS)
    assert ("full", "lab-c/model-new@high") in [(e.kind, e.subject) for e in out.events] or \
        any(e.kind == "gate_failed" for e in out.events)
    assert out.state.entry(HIGH).price_in == 120_000 and out.state.entry(HIGH).explore_pairs == 132
    # The list in force travels with the inputs, for a step that holds a
    # newer list back (the validator hands in the one `start` names).
    held = mi.step_inputs(start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE,
                          lab_list=mf.allowlist(3), lab_list_in_force=mf.ALLOW,
                          params=mf.PARAMS, category_of=CATS)
    assert held.lab_list_in_force == mf.ALLOW and held.lab_list.serial == 3


def test_inputs_in_any_order_are_the_same_inputs():
    start, epoch_state, window, proofs = _epoch_world()
    want = mi.step_inputs(start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE,
                          lab_list=mf.ALLOW, params=mf.PARAMS, category_of=CATS)
    rnd = random.Random(3)
    for _ in range(5):
        rnd.shuffle(window)
        rnd.shuffle(proofs)
        got = mi.step_inputs(start, epoch_state, window, proofs, epoch=100, nonce=mf.NONCE,
                             lab_list=mf.ALLOW, params=mf.PARAMS, category_of=dict(
                                 sorted(CATS.items(), key=lambda kv: rnd.random())))
        assert got == want


def test_step_inputs_need_the_scored_epochs_summary_and_menus():
    start, epoch_state, window, proofs = _epoch_world()
    kw = dict(epoch=100, nonce=mf.NONCE, lab_list=mf.ALLOW, params=mf.PARAMS, category_of=CATS)
    with pytest.raises(ValueError, match="no summary"):
        mi.step_inputs(start, epoch_state, window[:-1], proofs, **kw)
    with pytest.raises(ValueError, match="starts from"):
        mi.step_inputs(epoch_state, epoch_state, window, proofs, **kw)


def test_nothing_on_the_v1_to_v3_path_imports_this_module():
    import subprocess
    import sys
    code = ("import sys; import fugal_subnet.tee.verify, fugal_subnet.explore_pool, "
            "fugal_subnet.v2_epoch; print('fugal_subnet.menu_inputs' in sys.modules or "
            "'fugal_subnet.menu_state' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stdout + out.stderr
