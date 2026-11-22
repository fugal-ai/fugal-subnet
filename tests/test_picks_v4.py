"""Protocol v4, rule B as a one-epoch challenge (docs/decisions/SEAT_MINIMAL_V4.md
3.4, 3.7, 5.4, 8, as docs/decisions/FAST_CROWN_V4.md 2-4 changes them;
fugal_subnet/picks_v4.py, v2_epoch._challenge_epoch).

A challenger's item on each question of its prefix of the challenge list:
0 where its pick and the holder's are the same entry, else its value minus
the holder's, 100 x (grade - table cost) at $1 a correct answer
(1 / BENCHMARK_LAMBDA), each call at its entry's pick cost in the question's
category on THAT epoch's menu, failed calls failing closed; the one bound
with margin -1.5 points on at least 500 items, n0 100; a missed epoch
restarts the run. And the absolute exploration failure bound, waived in an
outage epoch.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet import config
from fugal_subnet import menu_inputs as mi
from fugal_subnet import menu_state as ms
from fugal_subnet import picks_v4 as pv
from fugal_subnet.seat_bound import lower_bound
from fugal_subnet.tee.proof import BenchmarkProof, QuestionResult
from tests import menu_fixtures as fx

HIGH, LOW, TWO = "lab-a/model-one@high", "lab-a/model-one@low", "lab-b/model-two"
U = config.BOUND_UNITS_PER_POINT_V4
HOLDER = "a1" * 32
OTHER = "b2" * 32


def _menu(*cells, epoch=101):
    """The fixture menu, HIGH carrying a pick-cost table built from `cells`."""
    s = fx.state(epoch=epoch)
    if not cells:
        return s
    table = ms.pick_table(cells, s.entry(HIGH), fx.PARAMS)
    return dataclasses.replace(s, entries=tuple(
        dataclasses.replace(e, pick=table) if e.id == HIGH else e for e in s.entries))


MENU = _menu(ms.PickCell("code", 10, 3_000, 25_000), ms.PickCell("math", 20, 4_000, 100_000))


def _v(state, entry, cat, grade):
    return pv.call_value(Fraction(grade), ms.pick_cost(state, entry, cat))


def _a(entry, correct=True, failed=False):
    return pv.Answer(entry, correct and not failed, failed)


# --- the constants ---------------------------------------------------------------------


def test_the_constants_are_the_designs():
    assert config.PICKS_MARGIN_POINTS_V4 == "3/2" and pv.PICKS_MARGIN_UNITS == -1_500_000
    assert config.PICKS_MIN_QUESTIONS_V4 == pv.PICKS_MIN_QUESTIONS == 500
    assert pv.BOUND_N0 == config.BOUND_N0_V4 == 100
    assert config.FAIL_MAX_BP_V4 == 3400 and config.SLICE_SIZE_V4 == 200
    assert pv.value_per_correct() == 1                 # $1: BENCHMARK_LAMBDA is 1


def test_v_is_read_from_benchmark_lambda_only(monkeypatch):
    from fugal_subnet import routing_protocol

    cost = Fraction(3, 1000)                          # three tenths of a cent
    assert pv.call_value(Fraction(1), cost) == 100 * U - 300_000
    monkeypatch.setattr(routing_protocol, "BENCHMARK_LAMBDA", 2.0)
    assert pv.value_per_correct() == Fraction(1, 2)
    assert pv.call_value(Fraction(1), cost) == 100 * U - 600_000
    monkeypatch.setattr(routing_protocol, "BENCHMARK_LAMBDA", 0.0)
    with pytest.raises(ValueError):
        pv.value_per_correct()


# --- the arithmetic, by hand ----------------------------------------------------------------


def test_rule_b_arithmetic_is_exact_on_a_hand_built_case():
    """Three questions, one challenge, every number worked by hand.

    q1 (code): the challenger picks HIGH (right), the holder TWO (right).
    q2 (math): the challenger's call to HIGH failed; the holder picks LOW
    (wrong). q3 (code): both pick HIGH: 0. V = $1, so a dollar is 100 points."""
    state = MENU
    items = pv.challenge_items(state, ["code", "math", "code"],
                               [_a(HIGH), _a(HIGH, failed=True), _a(HIGH)], [TWO, LOW, HIGH],
                               {0: _a(TWO), 1: _a(LOW, False)})
    t = {(e, c): ms.pick_cost(state, e, c) for e in (HIGH, LOW, TWO) for c in ("code", "math")}
    want = (round(100 * U * (1 - t[(HIGH, "code")])) - round(100 * U * (1 - t[(TWO, "code")])),
            round(100 * U * (0 - t[(HIGH, "math")])) - round(100 * U * (0 - t[(LOW, "math")])),
            0)
    assert items == want and all(type(x) is int for x in items)
    rec = pv.PicksEpoch(7, ("code", "math", "code"), {"H": items}, holder="B")
    got = pv.picks_items({7: rec}, "H", "B", (7,))
    assert got == {"code": [want[0], 0], "math": [want[1]]}
    # The NV of q1 in points, exactly: 100 x [(1 - 1) - (t_H - t_B) / V], within the one
    # rounding of each call's value.
    nv_q1 = 100 * (0 - (t[(HIGH, "code")] - t[(TWO, "code")]))
    assert abs(Fraction(got["code"][0], U) - nv_q1) <= Fraction(2, U)
    # The bound on those items is the one bound, at margin -1.5 points, minimum 500, n0 100.
    ev = pv.picks_check({7: rec}, "H", "B", through=7)
    assert ev.bound == lower_bound(got, margin_units=-1_500_000, n_min=500, n0=100)
    assert ev.paired == 3 and not ev.passes                 # 3 items, short of 500


def test_identical_picks_give_zero_and_the_sign_flips_with_the_roles():
    rng = np.random.default_rng(3)
    entries = [HIGH, LOW, TWO]
    cats = rng.choice(["code", "math", "hard"], size=300).tolist()
    mine = [entries[i] for i in rng.integers(0, 3, 300)]
    theirs = [entries[i] for i in rng.integers(0, 3, 300)]
    grades = rng.random((2, 300)) < 0.6
    own = [_a(e, bool(g)) for e, g in zip(mine, grades[0])]
    held = {i: _a(e, bool(g)) for i, (e, g) in enumerate(zip(theirs, grades[1]))
            if e != mine[i]}
    same = pv.challenge_items(MENU, cats, own, mine, {})
    assert same == (0,) * 300
    ab = pv.challenge_items(MENU, cats, own, theirs, held)
    back_own = [_a(e, bool(g)) for e, g in zip(theirs, grades[1])]
    back_held = {i: _a(mine[i], bool(grades[0][i])) for i in held}
    ba = pv.challenge_items(MENU, cats, back_own, mine, back_held)
    assert ab == tuple(-x for x in ba)
    recs = {e: pv.PicksEpoch(e, tuple(cats), {"A": same}, holder="A2") for e in range(10, 12)}
    ev = pv.picks_check(recs, "A", "A2", through=11)
    assert ev.paired == 600 and ev.bound.total == 0 and ev.passes    # 0 > -1.5, n >= 500


def test_failed_calls_fail_closed_at_their_table_cost():
    state = MENU
    assert pv.challenge_items(state, ["code"], [_a(HIGH, failed=True)], [None], {}) == (
        _v(state, HIGH, "code", 0),)
    assert pv.challenge_items(state, ["code"], [_a(HIGH)], [TWO], {0: _a(TWO, failed=True)}) == (
        _v(state, HIGH, "code", 1) - _v(state, TWO, "code", 1),)
    with pytest.raises(ValueError, match="no holder-side call"):
        pv.challenge_items(state, ["code"], [_a(HIGH)], [TWO], {})


def test_a_challenge_with_most_calls_failed_still_counts_and_never_gains():
    """No share of failed challenge calls voids an attempt: failures fail
    closed inside the items, so none can lift them."""
    n = 40
    own_ok = [_a(HIGH, i % 2 == 0) for i in range(n)]
    held_ok = {i: _a(TWO, i % 3 == 0) for i in range(n)}
    clean = pv.challenge_items(MENU, ["code"] * n, own_ok, [TWO] * n, held_ok)
    own_bad = [_a(HIGH, a.correct, failed=i < 30) for i, a in enumerate(own_ok)]
    held_bad = {i: _a(TWO, a.correct, failed=i >= 10) for i, a in held_ok.items()}
    worse = pv.challenge_items(MENU, ["code"] * n, own_bad, [TWO] * n, held_bad)
    assert all(w <= c for w, c in zip(worse, clean)) and sum(worse) < sum(clean)


def test_the_pick_cost_tables_cells_are_the_challengers_own_picks():
    """menu_inputs.graded_cells reads a proof's scored results: a challenge's
    own picks, never its holder-side calls."""
    own = [QuestionResult(f"q{i}", HIGH, True, 0.0, "ab" * 32, 100, 200) for i in range(4)]
    held = [QuestionResult(f"q{i}", TWO, True, 0.0, "ab" * 32, 100, 200, holder_pick=True)
            for i in range(2)]
    proof = _proof("5Fh", own + held)
    graded = mi.graded_cells([mi.VerifiedProof(proof, "ck")], MENU,
                             {f"q{i}": "code" for i in range(4)})
    assert {(g.entry_id, g.category): g.cells for g in graded} == {(HIGH, "code"): 4}


# --- the answered run --------------------------------------------------------------------


def _rec(e, heads, n=100, value=10 * U, cat="hard", holder="B"):
    return pv.PicksEpoch(e, (cat,) * n, {h: (value,) * n for h in heads}, holder=holder)


def test_a_missed_epoch_restarts_the_run():
    recs = {e: _rec(e, ("H",)) for e in range(1, 6)}
    assert pv.answered_run(recs, "H", 5) == (1, 2, 3, 4, 5)
    assert pv.picks_check(recs, "H", "B", through=5).passes            # 500 at +10
    recs[6] = _rec(6, ("X",))                                         # H misses epoch 6
    assert pv.answered_run(recs, "H", 6) == ()
    assert not pv.answered_in(recs, "H", 6) and pv.answered_in(recs, "X", 6)
    for e in range(7, 11):
        recs[e] = _rec(e, ("H",))
    assert pv.answered_run(recs, "H", 10) == (7, 8, 9, 10)            # restarted at 7
    ev = pv.picks_check(recs, "H", "B", through=10)
    assert ev.questions == ev.paired == 400 and not ev.passes          # short again
    recs[11] = _rec(11, ("H",))
    assert pv.picks_check(recs, "H", "B", through=11).passes


def test_an_epoch_nobody_was_verified_in_is_not_recorded_and_breaks_no_run():
    recs = {e: _rec(e, ("H",)) for e in (1, 2, 4, 5, 6)}              # epoch 3: no record
    assert pv.answered_run(recs, "H", 6) == (1, 2, 4, 5, 6)


def test_only_the_epochs_paired_with_the_bar_are_read():
    recs = {e: _rec(e, ("H",), holder="B" if e != 3 else "C") for e in range(1, 7)}
    ev = pv.picks_check(recs, "H", "B", through=6)
    assert ev.run == (1, 2, 3, 4, 5, 6) and ev.questions == 600
    assert ev.paired_epochs == 5 and ev.paired == 500 and ev.passes


def test_the_open_seat_asks_only_for_500_challenge_items():
    recs = {e: _rec(e, ("H",), holder=None) for e in range(1, 5)}
    ev = pv.picks_check(recs, "H", None, through=4)
    assert ev.bar is None and ev.bound is None and ev.lower is None
    assert ev.questions == 400 and not ev.passes
    recs[5] = _rec(5, ("H",), holder=None)
    assert pv.picks_check(recs, "H", None, through=5).passes
    # With a last holder it has 500 items paired with, the check is against it.
    worse = {e: _rec(e, ("H",), value=-5 * U, holder="L") for e in range(1, 6)}
    ev = pv.open_seat_check(worse, "H", "L", through=5)
    assert ev.bar == "L" and not ev.passes                        # 5 points worse
    few = {e: _rec(e, ("H",), value=-5 * U, holder="L" if e == 5 else None)
           for e in range(1, 6)}
    ev = pv.open_seat_check(few, "H", "L", through=5)
    assert ev.bar is None and ev.passes                           # 100 paired: count only


def test_the_holder_is_checked_against_nobody():
    recs = {e: _rec(e, ("B",)) for e in range(1, 6)}
    ev = pv.picks_check(recs, "B", "B", through=5)
    assert ev.bar is None and ev.questions == 500


# --- pricing at each epoch's own table ------------------------------------------------------


def test_each_epoch_is_priced_at_its_own_pick_cost_table():
    """A refresh of the pick-cost table between two epochs changes the price
    of the later epoch's calls only: the earlier epoch's items, computed at
    its own menu, are untouched."""
    before = _menu(ms.PickCell("code", 10, 3_000, 25_000), epoch=200)
    after = _menu(ms.PickCell("code", 40, 3_000, 900_000), epoch=201)
    assert ms.pick_cost(after, HIGH, "code") > ms.pick_cost(before, HIGH, "code")
    i200 = pv.challenge_items(before, ["code"], [_a(HIGH)], [None], {})
    i201 = pv.challenge_items(after, ["code"], [_a(HIGH)], [None], {})
    assert i200 == (_v(before, HIGH, "code", 1),) and i201 == (_v(after, HIGH, "code", 1),)
    assert i201[0] < i200[0]
    # A category the table does not list is priced at the entry's pick cost; an entry
    # with no table at its call cost; never at the call's own tokens.
    assert ms.pick_cost(before, HIGH, "knowledge") == ms.pick_cost(before, HIGH)
    assert ms.pick_cost(before, TWO, "code") == ms.costs(before)[TWO]


# --- the failure bound ---------------------------------------------------------------------


def _proof(hotkey, results, scored=True):
    return BenchmarkProof(epoch_id="e00000101", nonce="0f" * 32, questions_hash="1e" * 32,
                          weights_hash="2d" * 32, source_hash="3c" * 32, results=results,
                          total_cost_usd=0.0, per_model_costs={}, attestation_quote=b"",
                          timestamp=1.0, hotkey=hotkey, routing_protocol="ff" * 32,
                          menu_hash="cd" * 32, endpoint_reads=(), scored=scored)


def _explore(n, failed):
    return [QuestionResult(f"x{i}", HIGH, False, 0.0, "e3" * 32 if i < failed else "ab" * 32,
                           is_exploration=True, failed=i < failed) for i in range(n)]


@pytest.mark.parametrize("failed, refused", [(0, False), (34, False), (35, True), (100, True)])
def test_more_than_34_percent_of_exploration_failed_is_not_counted(failed, refused):
    honest = [mi.VerifiedProof(_proof(f"5Fh{k}", _explore(100, 1)), f"ck{k}") for k in range(4)]
    me = mi.VerifiedProof(_proof("5Fme", _explore(100, failed)), "ck-me")
    v = pv.failure_refusals(honest + [me], MENU)
    assert not v.outage and ("5Fme" in v.refused) is refused
    if refused:
        assert v.refused["5Fme"] == (failed, 100)


def test_an_outage_epoch_waives_the_bound_and_is_the_menu_steps_outage():
    many = [mi.VerifiedProof(_proof(f"5Fh{k}", _explore(100, 40)), f"ck{k}") for k in range(4)]
    v = pv.failure_refusals(many, MENU)
    summary = mi.EpochSummary(101, (), mi.exploration_calls(many, MENU), (), ())
    assert v.outage and mi.outage(summary) and v.refused == {}
    # Just under the outage rate: the median coldkey failed 30% exactly, not more.
    edge = [mi.VerifiedProof(_proof(f"5Fh{k}", _explore(100, 30 if k else 50)), f"ck{k}")
            for k in range(3)]
    v = pv.failure_refusals(edge, MENU)
    assert not v.outage and set(v.refused) == {"5Fh0"}


def test_one_operators_hotkeys_are_one_voice_in_the_outage():
    """Twenty failing hotkeys of one coldkey do not make an outage that
    waives the bound for themselves."""
    honest = [mi.VerifiedProof(_proof(f"5Fh{k}", _explore(100, 2)), f"ck{k}") for k in range(3)]
    junk = [mi.VerifiedProof(_proof(f"5Fj{k:02d}", _explore(100, 90)), "ck-junk")
            for k in range(20)]
    v = pv.failure_refusals(honest + junk, MENU)
    assert not v.outage and set(v.refused) == {f"5Fj{k:02d}" for k in range(20)}


# --- the published form ---------------------------------------------------------------------


def test_the_record_round_trips_through_the_reveal_and_the_window_file(tmp_path):
    from fugal_subnet.champion import Window
    from fugal_subnet.v2_epoch import build_epoch_slice, load_window, save_window

    cats = ("hard", "code") * 500
    rec = pv.PicksEpoch(4, cats, {OTHER: tuple(range(-250, 250)),
                                  "c3" * 32: tuple([2**31 - 1, -(2**31)] * 500)}, True, HOLDER)
    qs = [f"q{i}" for i in range(1000)]
    block = pv.reveal_block(rec, {"A": {"x": 1}}, {"5Fx": (40, 100)}, questions=qs)
    assert list(block["items"]) == sorted(rec.values) and block["failure_refused"] == {
        "5Fx": [40, 100]}
    assert block["holder"] == HOLDER and block["list"][0] == ["q0", "hard"]
    back = pv.epoch_from_reveal(block, 4)
    assert back == dataclasses.replace(rec, values=dict(sorted(rec.values.items())))
    with pytest.raises(ValueError):
        pv.epoch_from_reveal({**block, "items": {OTHER: block["items"][OTHER][:-8]}}, 4)
    with pytest.raises(ValueError):
        pv.epoch_from_reveal({**block, "outage": 1}, 4)
    sq = [{"question_id": "a", "benchmark": "x"}, {"question_id": "b", "benchmark": "y"}]
    sl = build_epoch_slice(4, sq, [HIGH], {}, {}, {HIGH: (0.0, 0.0)},
                           {"x": "hard", "y": "code"})
    w = Window.empty(48).add_epoch(sl, {}, picks=rec)
    path = str(tmp_path / "w.npz")
    save_window(path, w)
    assert load_window(path, 48).picks == w.picks
    # A v1-v3 window keeps no picks and its file has no picks key.
    save_window(path, Window.empty(48).add_epoch(sl, {}))
    assert load_window(path, 48).picks == {}
    with pytest.raises(ValueError, match="this epoch"):
        Window.empty(48).add_epoch(sl, {}, picks=dataclasses.replace(rec, epoch=5))


# --- the note's pass rates, by simulation ------------------------------------------------------

# FAST_CROWN_V4.md 2: one attempt judged alone (n0 = 100), share passed at a
# true NV of -1 / 0 / +1 / +2 points. The note resamples measured items (SD
# 20, heavy tails); this test draws normal items of SD 20, which pass a
# little more often near the margin, so its tolerance is eight points.
DESIGN_PASS_RATES = {500: (0.0, 0.023, 0.18, 0.57), 1000: (0.001, 0.078, 0.56, 0.95)}
TOLERANCE = 0.08


def _simulate(n, runs, seed, sd=20.0):
    from fugal_subnet.seat_bound import Stratum, bound_from_strata

    rng = np.random.default_rng(seed)
    out = []
    for d in (-1.0, 0.0, 1.0, 2.0):
        ok = 0
        for _ in range(runs):
            xs = np.rint(rng.normal(d, sd, n) * U).astype(np.int64).tolist()
            ok += bound_from_strata({"all": Stratum(n, sum(xs), sum(v * v for v in xs))},
                                    margin_units=pv.PICKS_MARGIN_UNITS,
                                    n_min=pv.PICKS_MIN_QUESTIONS, n0=pv.BOUND_N0).passes
        out.append(ok / runs)
    return tuple(out)


@pytest.mark.parametrize("n", sorted(DESIGN_PASS_RATES))
def test_the_notes_pass_rates_are_reproduced_by_simulation(n):
    got = _simulate(n, runs=500, seed=7)
    want = DESIGN_PASS_RATES[n]
    assert all(abs(g - w) <= TOLERANCE for g, w in zip(got, want)), (n, got, want)


# --- one epoch end to end: the recomputed picks, the record, the reveal --------------------


DIM = 8
N = 500


def _head(seed, rows=(HIGH, LOW, TWO)):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((len(rows), DIM)) * 2.0, rng.standard_normal(len(rows)),
            list(rows))


def _hidden(qid):
    v = np.random.default_rng(abs(hash(qid)) % 2**32).standard_normal(DIM)
    return v / np.linalg.norm(v)


class _Epoch:
    """One epoch's challenge check (v2_epoch._challenge_epoch) on the
    fixture menu: the holder named for it, heads by UID, a challenge list
    of N questions."""

    def __init__(self, epoch=12, holder=HOLDER, holder_head=None):
        from fugal_subnet import seat_v4

        self.epoch, self.holder = epoch, holder
        self.qids = [f"c{epoch}-{i}" for i in range(N)]
        self.cats = [("code", "math")[i % 2] for i in range(N)]
        self.cache = {q: _hidden(q) for q in self.qids}
        self.heads = {9: holder_head or _head(1)}             # the holder serves too
        self.hash_of = {9: holder}
        self.seat = seat_v4.SeatStateV4(holders=((epoch - 2, holder),))

    def picks(self, head, n=N):
        from fugal_subnet.head_odds import picks_and_odds_v4
        from fugal_subnet.v2_epoch import _artifact

        return list(picks_and_odds_v4(_artifact(head), MENU,
                                      np.stack([self.cache[q] for q in self.qids[:n]])).picks)

    def proof(self, uid, head, *, n=N, holder=None, own=None, held=None):
        """A challenge proof with the picks the heads give (unless `own` /
        `held` override them): every own call right, every holder call wrong."""
        holder = self.holder if holder is None else holder
        mine = own or self.picks(head, n)
        theirs = self.picks(self.heads[9], n)
        held = held if held is not None else {
            q: hp for q, m, hp in zip(self.qids, mine, theirs) if hp is not None and hp != m}
        self.heads[uid], self.hash_of[uid] = head, f"{uid:02x}" * 32
        res = [SimpleNamespace(question_id=q, routed_model=m, correct=True, failed=False)
               for q, m in zip(self.qids, mine)]
        hres = [SimpleNamespace(question_id=q, routed_model=e, correct=False, failed=False)
                for q, e in held.items()]
        return SimpleNamespace(scored=True, challenge=SimpleNamespace(size=n, holder=holder),
                               scored_results=res, holder_results=hres)

    def run(self, proofs):
        from fugal_subnet.champion import Window
        from fugal_subnet.v2_epoch import PicksInput, _challenge_epoch

        proofs = {9: SimpleNamespace(scored=False, challenge=None, scored_results=[],
                                     holder_results=[]), **proofs}
        serving = {self.hash_of[u]: u for u in proofs}
        oi = SimpleNamespace(heads=self.heads, hidden_of=lambda q: self.cache[q])
        sl = SimpleNamespace(question_ids=tuple(self.qids[:200]))
        pi = PicksInput(state=MENU, challenge=tuple(zip(self.qids, self.cats)))
        return _challenge_epoch(pi, self.epoch, sl, serving, proofs, self.seat,
                                Window.empty(48), oi)[:2]


def test_a_challenge_with_the_heads_picks_is_counted_at_its_items():
    ep = _Epoch()
    head = _head(2)
    rec, block = ep.run({1: ep.proof(1, head)})
    h = ep.hash_of[1]
    mine, theirs = ep.picks(head), ep.picks(ep.heads[9])
    held = {i: pv.Answer(t, False, False) for i, (m, t) in enumerate(zip(mine, theirs)) if t != m}
    want = pv.challenge_items(MENU, ep.cats, [pv.Answer(m, True, False) for m in mine], theirs,
                              held)
    assert rec.values == {h: want} and rec.holder == HOLDER
    assert block["refused"] == {}
    assert [q for q, _c in block["list"]] == ep.qids
    assert pv.epoch_from_reveal(block, ep.epoch) == rec
    assert block["evidence"][h]["answered"] is True
    assert block["evidence"][ep.hash_of[9]]["answered"] is False      # the holder: no challenge
    assert sum(n for n, _t in block["item_means"].values()) == len(held)


def test_a_challenge_that_states_other_picks_or_another_holder_is_refused():
    head = _head(2)
    cases = {}
    ep = _Epoch()
    mine = ep.picks(head)
    other = [TWO if m != TWO else HIGH for m in mine]
    cases["states a pick other than its head's"] = ep.proof(1, head, own=other)
    ep2 = _Epoch()
    cases["pairs with a holder other than the one this epoch names"] = ep2.proof(
        1, head, holder=OTHER)
    ep3 = _Epoch()
    p = ep3.proof(1, head)
    p.holder_results = p.holder_results[:-1]                       # one holder call missing
    cases["states a holder pick other than the holder's"] = p
    for (why, proof), ep_ in zip(cases.items(), (ep, ep2, ep3)):
        rec, block = ep_.run({1: proof})
        assert rec.values == {} and block["refused"] == {ep_.hash_of[1]: why}, why


def test_without_the_holders_head_the_epoch_is_not_scored_here():
    from fugal_subnet.v2_epoch import HolderHeadMissing

    ep = _Epoch()
    p = ep.proof(1, _head(2))
    del ep.heads[9]
    ep.hash_of[9] = "dd" * 32                                      # someone else serves
    with pytest.raises(HolderHeadMissing):
        ep.run({1: p})


def test_with_no_holder_the_challengers_side_is_scored_alone():
    from fugal_subnet import seat_v4

    ep = _Epoch()
    ep.seat = seat_v4.SeatStateV4()                                # no holder recorded
    head = _head(2)
    p = ep.proof(1, head, holder=None, held={})
    p.challenge.holder = None
    rec, block = ep.run({1: p})
    mine = ep.picks(head)
    assert rec.holder is None
    assert rec.values[ep.hash_of[1]] == pv.challenge_items(
        MENU, ep.cats, [pv.Answer(m, True, False) for m in mine], [None] * N, {})


def test_a_v3_epoch_has_no_picks_and_its_reveal_is_unchanged():
    from tests import test_v2_epoch as t

    r = t.run(0, {1: t.oracle, 2: t.always("m/mid")})
    assert "picks_v4" not in r.reveal and pv.DIGEST_KEY not in r.reveal["v2_state"]
    assert r.window.picks == {}


@pytest.mark.parametrize("protocol, kept", [(3, None), (4, False)])
def test_a_rebuild_reads_an_exploration_only_head_only_from_a_v4_reveal(protocol, kept):
    """A head serving with no scored answers (an empty routing row) answered
    nothing in a v4 reveal; in a v1-v3 reveal the row is malformed, refused
    by name as before."""
    from fugal_subnet.v2_rebuild import epoch_from_reveal

    reveal = {"epoch_id": "e00000050", "models": [HIGH], "protocol_version": protocol,
              "questions": [{"question_id": "q1", "benchmark": "x"}],
              "shared_grades": [{"question_id": "q1", "model": HIGH, "grade": 1.0}],
              "failed": {}, "routing_decisions": {"1": [], "2": [0]},
              "v2_state": {"serving": {"h1": 1, "h2": 2},
                           "rules": {"protocol": protocol, "seat": "odds"}}}
    if kept is None:
        with pytest.raises(ValueError, match="one model index per question"):
            epoch_from_reveal(reveal, {HIGH: (0.0, 0.0)})
        return
    _sl, answers = epoch_from_reveal(reveal, {HIGH: (0.0, 0.0)})
    assert set(answers) == {"h2"}
