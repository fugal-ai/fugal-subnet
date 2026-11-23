"""Protocol v4: a challenge's holder side (FAST_CROWN_V4.md 2, as INVARIANTS
"I3 - v4: rule B's items" now reads it).

- Picks are computed over the menu's FULL entries only, on both sides, in
  the TD, on every node and in the published `holder_picks`: a provisional
  entry is never a challenge pick.
- A holder-side call the challenger's TD did not start (its time or dollar
  budget ran out) is no item; one made that failed scores as the holder
  answering right; when more than 10% of the holder-side calls failed or were
  not made, the challenge is void: counted for and against nobody, the reason
  published, and the run passes through it (once).
- The TD starts the challenger's own calls first, so slow holder-side calls
  cannot starve them, and reports what was not started instead of dropping
  the challenge.

No network, no spend.
"""
from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet import menu_state, picks_v4
from fugal_subnet.tee import harness
from fugal_subnet.tee.proof import BenchmarkProof, Challenge, QuestionResult
from fugal_subnet.tee.runtime import APICallRecord
from tests import menu_fixtures as fx

CAT = "math"
N = 1000
H1, H2 = (hashlib.sha256(x).hexdigest() for x in (b"holder", b"challenger"))
A = picks_v4.Answer


@pytest.fixture
def small_menus(monkeypatch):
    fx.allow_small_menus(monkeypatch)


def _state():
    state = fx.provisional(fx.state(epoch=200))
    new = next(e.id for e in state.entries if e.provisional)
    return state, new


def _favouring(state, favour: str, dim: int = 4):
    ids = list(state.ids())
    return SimpleNamespace(W=np.zeros((len(ids), dim), np.float32),
                           b=np.array([4.0 if m == favour else -4.0 + 0.1 * i
                                       for i, m in enumerate(ids)], np.float32), models=ids)


# --- (a) picks over full entries only -------------------------------------------------------


def test_a_provisional_entry_is_never_a_challenge_pick_on_either_side(small_menus):
    state, new = _state()
    hidden = np.eye(4, dtype=np.float32)[:3]
    picks = picks_v4.challenge_picks(_favouring(state, new), state, hidden)
    assert new not in picks and all(not state.entry(p).provisional for p in picks)
    # a head with rows for the provisional entry only has no challenge pick
    only = SimpleNamespace(W=np.zeros((1, 4), np.float32), b=np.zeros(1, np.float32),
                           models=[new])
    assert picks_v4.challenge_picks(only, state, hidden) == (None, None, None)


# --- the reviewer's scenario: a holder whose differing picks do not complete ----------------


def _scenario(state, holder_entry, own_entry, fail_share, *, not_started, seed=1):
    """Challenger right 90%, holder 50% on its differing picks (87% of the
    questions); `fail_share` of the holder-side calls not started
    (`not_started`) or made and failed."""
    rng = random.Random(seed)
    own, hp, held = [], [], {}
    for i in range(N):
        own.append(A(own_entry, rng.random() < 0.90, False))
        if rng.random() < 0.87:
            hp.append(holder_entry)
            bad = rng.random() < fail_share
            held[i] = A(holder_entry, (not bad) and rng.random() < 0.50, bad, not (bad and not_started))
        else:
            hp.append(own_entry)
    return own, hp, held


def _judge(state, own, hp, held):
    why = picks_v4.challenge_verdict(held)
    items = (picks_v4.void_items(N) if why else
             picks_v4.challenge_items(state, [CAT] * N, own, hp, held))
    rec = {7: picks_v4.PicksEpoch(7, (CAT,) * N, {"chal": items}, False, "holder")}
    return why, items, picks_v4.picks_check(rec, "chal", "holder", through=7), rec


def test_a_much_better_challenger_passes_when_the_holders_calls_are_answered(small_menus):
    state, _new = _state()
    full = [m for m in state.ids() if not state.entry(m).provisional]
    own, hp, held = _scenario(state, full[1], full[0], 0.0, not_started=False)
    why, items, ev, _ = _judge(state, own, hp, held)
    assert why is None and ev.passes and sum(items) > 30 * N * picks_v4.UNITS_PER_POINT


@pytest.mark.parametrize("not_started", [True, False])
def test_holder_calls_mostly_not_made_or_failed_void_the_challenge_and_keep_the_run(
        small_menus, not_started):
    """706 of 866 holder-side calls not made (the budget-cut case), or every
    one failed (a host down): the challenge counts for nobody, says why, and
    the challenger's earlier counted epoch still stands in its run."""
    state, _new = _state()
    full = [m for m in state.ids() if not state.entry(m).provisional]
    for share in (706 / 866, 1.0):
        own, hp, held = _scenario(state, full[1], full[0], share, not_started=not_started)
        why, items, ev, rec = _judge(state, own, hp, held)
        assert why and "holder-side calls failed or were not made" in why
        assert picks_v4.is_void(items) and not ev.passes and ev.questions == 0
        assert not picks_v4.answered_in(rec, "chal", 7)
        good = picks_v4.PicksEpoch(6, (CAT,) * N, {"chal": (2_000_000,) * N}, False, "holder")
        both = {6: good, 7: rec[7]}
        assert picks_v4.answered_run(both, "chal", 7) == (6, 7)
        assert picks_v4.picks_check(both, "chal", "holder", through=7).paired == N


def test_failed_holder_calls_up_to_the_cap_score_as_right_and_unmade_ones_are_no_items(
        small_menus):
    state, _new = _state()
    full = [m for m in state.ids() if not state.entry(m).provisional]
    o, h = full[0], full[1]
    right = picks_v4.challenge_items(state, [CAT], [A(o, False, False)], [h], {0: A(h, True, False)})
    failed = picks_v4.challenge_items(state, [CAT], [A(o, False, False)], [h],
                                      {0: A(h, False, True)})
    unmade = picks_v4.challenge_items(state, [CAT], [A(o, False, False)], [h],
                                      {0: A(h, False, True, False)})
    assert failed == right and unmade == (picks_v4.NO_ITEM,)
    # exactly at the cap the challenge counts; one more call and it is void
    held = {i: A(h, True, False) for i in range(100)}
    for i in range(10):
        held[i] = A(h, False, True, i % 2 == 0)
    assert picks_v4.challenge_verdict(held) is None
    held[10] = A(h, False, True)
    assert picks_v4.challenge_verdict(held)


def test_a_run_passes_through_one_void_epoch_only():
    void = picks_v4.void_items(500)
    good = (1_000_000,) * 500

    def rec(e, items):
        return picks_v4.PicksEpoch(e, ("a",) * 1000, {"c": items}, holder=H1)
    recs = {1: rec(1, good), 2: rec(2, void), 3: rec(3, good)}
    assert picks_v4.answered_run(recs, "c", 3) == (1, 2, 3)
    assert picks_v4.picks_check(recs, "c", H1, through=3).paired == 1000
    recs[4] = rec(4, void)
    recs[5] = rec(5, good)
    # the second void back from the newest ends the run there
    assert picks_v4.answered_run(recs, "c", 5) == (3, 4, 5)
    assert picks_v4.picks_check(recs, "c", H1, through=5).paired == 1000


def test_no_real_item_is_the_no_item_mark_and_item_means_skip_it():
    assert picks_v4._clamp32(-(2**40)) != picks_v4.NO_ITEM
    means = picks_v4.item_means((5, picks_v4.NO_ITEM, 0), ["x", "x", "y"], ["h", "h", "y"])
    assert means == {"x": (1, 5)}


def test_the_reveal_publishes_void_challenges_and_reads_them_back_strictly():
    items = {H2: picks_v4.void_items(500), H1: (3,) * 500}
    rec = picks_v4.PicksEpoch(4, ("math",) * 1000, items, False, H1)
    block = picks_v4.reveal_block(rec, {}, void={H2: "why"},
                                  questions=[f"q{i}" for i in range(1000)])
    assert block["void"] == {H2: "why"}
    back = picks_v4.epoch_from_reveal(block, 4)
    assert back.values == rec.values
    bad = dict(block, void={})
    with pytest.raises(ValueError, match="disagree"):
        picks_v4.epoch_from_reveal(bad, 4)
    bad = dict(block, void={H2: "why", H1: "why"})
    with pytest.raises(ValueError, match="disagree"):
        picks_v4.epoch_from_reveal(bad, 4)


# --- the proof: which holder-side calls were not started --------------------------------------


def _result(qid, entry, *, holder=None, not_started=None):
    failed = bool(not_started)
    return QuestionResult(question_id=qid, routed_model=entry, correct=not failed,
                          cost_usd=0 if failed else 0.001, response_hash="0" * 64,
                          prompt_tokens=0 if failed else 1, completion_tokens=0 if failed else 1,
                          is_exploration=False, latency_ms=0, provider_cost=None, served_by="",
                          holder_pick=holder, failed=failed, not_started=not_started)


def _proof(results):
    return BenchmarkProof(epoch_id="e", nonce="0" * 64, questions_hash="0", weights_hash="0",
                          source_hash="0", results=results, total_cost_usd=0.0,
                          per_model_costs={}, attestation_quote=b"", timestamp=1.0,
                          hotkey="hk", routing_protocol="fugal-v4:x", menu_hash="1" * 64,
                          endpoint_reads=(), scored=True, challenge=Challenge(500, H1))


def test_not_started_round_trips_is_hashed_and_only_on_a_failed_holder_call():
    seed = menu_state.load_seed()
    a, b = seed.ids()[:2]
    p = _proof([_result("q1", a), _result("q1", b, holder=True, not_started=True)])
    back = BenchmarkProof.from_dict(p.to_dict())
    assert back.content_hash() == p.content_hash() and back.results[1].not_started is True
    plain = _proof([_result("q1", a), _result("q1", b, holder=True)])
    assert plain.content_hash() != p.content_hash()
    assert "not_started" not in plain.to_dict()["results"][1]
    d = p.to_dict()
    d["results"][0]["not_started"] = True
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)
    d = p.to_dict()
    d["results"][1]["completion_tokens"] = 5
    with pytest.raises(ValueError):
        BenchmarkProof.from_dict(d)


def test_verification_holds_the_not_started_calls_to_the_end_of_the_tds_order():
    from fugal_subnet.tee.verify import V4Expected, _v4_scored_check

    seed = menu_state.load_seed()
    a, b = seed.ids()[:2]
    lst = [f"q{i}" for i in range(1000)]
    exp = V4Expected(state=seed, menu_hash="1" * 64, hotkey="hk", shared={}, cost_aware={},
                     coverage={}, challenge=tuple(lst))
    own = [_result(q, a) for q in lst[:500]]
    nonce = bytes.fromhex("0" * 64)
    order = sorted(lst[:40], key=lambda q: picks_v4.call_order_key(nonce, q))

    def held(cut):
        return [_result(q, b, holder=True, not_started=True if q in cut else None)
                for q in order]
    sl = set(lst[:200])
    assert _v4_scored_check(_proof(own + held(set(order[-7:]))), exp, sl) is None
    assert _v4_scored_check(_proof(own + held(set(order))), exp, sl) is None
    why = _v4_scored_check(_proof(own + held({order[3]})), exp, sl)
    assert why and "not the last" in why
    stray = own[:-1] + [_result(lst[499], a, not_started=True)]
    stray[-1] = QuestionResult(**{**stray[-1].__dict__, "holder_pick": None})
    assert _v4_scored_check(_proof(stray), exp, sl)


# --- the TD: own calls first, the budget cuts holder-side calls, nothing dropped -------------

SCALE = 200.0


def _timed_run(holder_median_s, holder_sigma, *, seed=1, n=N, differ=0.87, budget_s=1800):
    """The shipped call loop (`_call_all_v4`: 40 in flight, the budget) in the
    harness's order, on a scaled clock with lognormal latencies."""
    rng = random.Random(seed)
    lat = {}
    req = {"model": "m/x", "max_tokens": 10, "provider": {"order": ["h"], "allow_fallbacks": False}}
    own, rest = [], []
    for i in range(n):
        q = {"question_id": f"q{i}"}
        own.append((q, "own/fast", "scored", req, "own/fast"))
        lat[(f"q{i}", "scored")] = rng.lognormvariate(math.log(8.0), 0.8)
        if rng.random() < differ:
            rest.append((q, "slow/entry", "holder", req, "slow/entry"))
            lat[(f"q{i}", "holder")] = rng.lognormvariate(math.log(holder_median_s), holder_sigma)
    for i in range(112):
        q = {"question_id": f"x{i}"}
        rest.append((q, "own/fast", "explore", req, "own/fast"))
        lat[(f"x{i}", "explore")] = rng.lognormvariate(math.log(10), 0.8)
    nonce = b"\x01" * 32
    order = own + sorted(rest, key=lambda it: picks_v4.call_order_key(nonce, it[0]["question_id"]))
    t0 = time.monotonic()

    def clock():
        return (time.monotonic() - t0) * SCALE
    starts, lock = {}, threading.Lock()
    proxy = SimpleNamespace(records=[], lock=threading.Lock())

    def call(proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        rid = harness.current_request_id()
        kind = "holder" if rid.endswith(":h") else "scored" if rid.endswith(":c") else "explore"
        with lock:
            starts[(question["question_id"], kind)] = clock()
        time.sleep(lat[(question["question_id"], kind)] / SCALE)
        with proxy.lock:
            proxy.records.append(APICallRecord(model_id=model_id, prompt_tokens=1,
                                               completion_tokens=1, cost_usd=0, timestamp=0.0,
                                               response_hash="", request_id=rid))
        return "ok"
    return order, starts, call, proxy, clock, budget_s


def test_slow_holder_calls_never_starve_the_challengers_own_and_are_cut_at_the_budget(
        monkeypatch):
    order, starts, call, proxy, clock, budget = _timed_run(40.0, 1.0)
    monkeypatch.setattr(harness, "_call_model", call)
    out = harness._call_all_v4(proxy, order, "e1", budget_s=budget, clock=clock)
    kinds = [it[2] for it in order]
    own_started = [not o.not_started for k, o in zip(kinds, out) if k == "scored"]
    assert all(own_started)
    assert harness.reported_size(1000, own_started) == 1000
    held = [o.not_started for k, o in zip(kinds, out) if k == "holder"]
    # the calls not started are the last of the holder-side order, and none
    # started past the budget (a little scheduling slack on the scaled clock)
    assert held == sorted(held)
    assert all(t <= budget + 60 for (_q, k), t in starts.items() if k != "explore")
    # exploration is never cut
    assert not any(o.not_started for k, o in zip(kinds, out) if k == "explore")


def test_own_calls_cut_report_the_largest_size_whose_calls_were_all_made():
    assert harness.reported_size(1000, [True] * 1000) == 1000
    assert harness.reported_size(1000, [True] * 700 + [False] * 300) == 500
    assert harness.reported_size(1000, [True] * 499 + [False] * 501) is None
    assert harness.reported_size(500, [True] * 500) == 500


def test_the_challenge_ordering_starts_calls_in_order_even_with_many_workers(monkeypatch):
    """The start decisions are made in the given order: a call never starts
    after a later one was refused."""
    t = [0.0]
    req = {"model": "m/x", "max_tokens": 10, "provider": {"order": ["h"], "allow_fallbacks": False}}
    items = [({"question_id": f"q{i}"}, "e", "holder", req, "e") for i in range(400)]

    def call(proxy, model_id, question, request=None, reasoning=None, deadline_s=None):
        t[0] += 1.0
        time.sleep(0.0005)
        return "ok"
    monkeypatch.setattr(harness, "_call_model", call)
    out = harness._call_all_v4(SimpleNamespace(records=[]), items, "e", budget_s=150,
                               clock=lambda: t[0])
    flags = [o.not_started for o in out]
    assert flags == sorted(flags) and any(flags) and not all(flags)

