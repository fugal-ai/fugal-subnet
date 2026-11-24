"""Protocol v5's exam (exam_v5): the draw, the cells only where comparisons
need them, votes from distinct coldkeys, and paired items no compared
operator can move."""
from __future__ import annotations

import io
import random
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import arena_score, config, exam_v5
from fugal_subnet.exam_v5 import ExamSpec, Vote, build_spec
from fugal_subnet.head_eval import load_head_v4
from fugal_subnet.head_odds import picks_and_odds_v4
from tests import menu_fixtures as mf
from tests.test_head_v4 import _arrays, _hidden

NONCE = bytes(range(32))
Y = config.Y_SCALE_V5
MENU = ["m/a", "m/b", "m/c", "m/d"]
RING = [(f"hk{i}", f"ck{i % 6}") for i in range(12)]


def _spec(contenders, pool=None, ring=RING, baseline="m/a", epoch=7):
    pool = pool or [f"q{i}" for i in range(len(next(iter(contenders.values()))))]
    return build_spec(epoch, NONCE, contenders, MENU, pool, ring, baseline)


# --- the draw ------------------------------------------------------------------


def test_draw_takes_each_datasets_quota_in_keyed_order_and_is_deterministic():
    by = {"mmlupro": [f"p{i}" for i in range(500)], "trivia": [f"t{i}" for i in range(500)],
          "empty": []}
    w = {"mmlupro": 600_000, "trivia": 400_000, "empty": 100_000, "zero": 0}
    d = exam_v5.draw_exam(NONCE, by, w, 100)
    assert len(d) == 100 and len(set(d)) == 100
    assert sum(q.startswith("p") for q in d) == 60
    rev = {k: list(reversed(v)) for k, v in reversed(list(by.items()))}
    assert exam_v5.draw_exam(NONCE, rev, w, 100) == d
    assert exam_v5.draw_exam(bytes(32), by, w, 100) != d
    assert exam_v5.draw_exam(NONCE, by, w, 0) == []


def test_draw_quota_by_largest_remainder_and_short_datasets_stay_short():
    by = {"a": ["a1", "a2"], "b": [f"b{i}" for i in range(10)], "c": [f"c{i}" for i in range(10)]}
    d = exam_v5.draw_exam(NONCE, by, {"a": 1, "b": 1, "c": 1}, 10)
    counts = {k: sum(q.startswith(k) for q in d) for k in "abc"}
    assert counts == {"a": 2, "b": 3, "c": 3}  # a's quota 4 (3+1) capped at 2; no refill


# --- picks at the head's own lam -----------------------------------------------


def _head(models, lam, seed=0):
    z = _arrays(models, seed=seed)
    z["lam"] = np.float64(lam)
    buf = io.BytesIO()
    np.savez(buf, **z)
    return load_head_v4(buf.getvalue())


def test_exam_picks_use_the_heads_own_lam():
    state = mf.state()
    ids = list(state.ids())
    hidden = _hidden(40)
    cheap = _head(ids, 0.001)
    picks = exam_v5.exam_picks(cheap, state, hidden)
    assert picks == tuple(p or "" for p in
                          picks_and_odds_v4(cheap, state, hidden, lam=Fraction(0.001)).picks)
    dear = _head(ids, 9000.0)
    assert exam_v5.exam_picks(dear, state, hidden) == tuple(
        p or "" for p in picks_and_odds_v4(dear, state, hidden, lam=Fraction(9000.0)).picks)


@pytest.mark.parametrize("lam", [0.0, -1.0, float("inf"), float("nan"), 10_001.0])
def test_a_head_lam_outside_the_bounds_is_refused(lam):
    with pytest.raises(ValueError):
        exam_v5.head_lam(_head(list(mf.state().ids()), lam))


# --- the spec ------------------------------------------------------------------


def test_cells_only_where_picks_differ_and_baseline_cells_on_the_first_questions(monkeypatch):
    monkeypatch.setattr(exam_v5, "EXAM_BASELINE_QUESTIONS", 2)
    # q0: all equal to B* -> none; q1: equal, not B* -> B* cells (baseline q);
    # q2: equal, not B*, past the baseline questions -> none; q3: differ -> both.
    s = _spec({"h1": ["m/a", "m/b", "m/b", "m/c"], "h2": ["m/a", "m/b", "m/b", "m/d"]})
    assert s.questions == ("q0", "q1", "q2", "q3")
    assert s.cells == (("q1", "m/a"), ("q1", "m/b"), ("q3", "m/c"), ("q3", "m/d"))
    assert s.baseline_questions == ("q0", "q1")


def test_an_off_menu_pick_is_no_pick_and_a_no_pick_needs_no_cell():
    s = _spec({"h1": ["x/off", "m/b"], "h2": ["m/a", "m/b"]}, baseline="")
    assert s.picks["h1"] == ("", "m/b")
    assert s.cells == (("q0", "m/a"),)
    assert s.baseline == "" and s.baseline_questions == ()


def test_each_cell_gets_votes_from_distinct_coldkeys_and_no_hotkey_exceeds_its_budget(monkeypatch):
    monkeypatch.setattr(exam_v5, "EXAM_CALLS_PER_HOTKEY", 5)
    rng = random.Random(3)
    picks = {f"h{i}": [rng.choice(MENU) for _ in range(200)] for i in range(5)}
    s = _spec(picks)
    ck = dict(RING)
    load: dict[str, int] = {}
    for cell in s.cells:
        hks = s.assignment[cell]
        assert len(hks) == config.EXAM_VOTES_V5
        assert len({ck[h] for h in hks}) == len(hks)
        for h in hks:
            load[h] = load.get(h, 0) + 1
    assert max(load.values()) <= 5
    assert len(s.cells) * config.EXAM_VOTES_V5 <= 5 * len(RING)
    assert sum(len(exam_v5.assigned_cells(s, h)) for h, _ in RING) == len(s.cells) * 3


def test_the_budget_stops_the_exam_in_order_and_the_cap_holds(monkeypatch):
    monkeypatch.setattr(exam_v5, "EXAM_CALLS_PER_HOTKEY", 1)    # 12 calls: 4 two-cell questions
    picks = {"h1": ["m/a"] * 50, "h2": ["m/b"] * 50}
    s = _spec(picks, baseline="")
    assert s.questions == ("q0", "q1")       # 2 cells x 3 votes = 6 per question
    monkeypatch.setattr(exam_v5, "EXAM_CALLS_PER_HOTKEY", 10_000)
    monkeypatch.setattr(exam_v5, "EXAM_QUESTIONS", 7)
    assert len(_spec(picks, baseline="").questions) == 7


def test_equal_picks_cost_nothing_so_the_exam_keeps_going():
    picks = {"h1": ["m/c"] * 600, "h2": ["m/c"] * 600}
    s = _spec(picks, baseline="m/c")
    assert len(s.questions) == config.EXAM_QUESTIONS_V5 and s.cells == ()


def test_too_few_coldkeys_give_fewer_votes():
    s = _spec({"h1": ["m/a"], "h2": ["m/b"]}, ring=[("hk1", "ck1"), ("hk2", "ck2")], baseline="")
    assert all(len(v) == 2 for v in s.assignment.values())
    empty = _spec({"h1": ["m/a"], "h2": ["m/b"]}, ring=[], baseline="")
    assert empty.questions == ()


def test_the_spec_is_canonical_and_independent_of_input_order():
    rng = random.Random(1)
    picks = {f"h{i}": [rng.choice(MENU) for _ in range(120)] for i in range(4)}
    a = _spec(picks)
    b = build_spec(7, NONCE, dict(reversed(list(picks.items()))), list(reversed(MENU)),
                   [f"q{i}" for i in range(120)], list(reversed(RING)), "m/a")
    assert a.to_json() == b.to_json() and a.spec_hash() == b.spec_hash()
    back = ExamSpec.from_json(a.to_json())
    assert back.to_json() == a.to_json()
    with pytest.raises(ValueError):
        ExamSpec.from_json(a.to_json().replace('{"assignment"', '{ "assignment"'))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        _spec({"h1": ["m/a", "m/b"]}, pool=["q0"])
    with pytest.raises(ValueError):
        _spec({"h1": ["m/a", "m/b"]}, pool=["q0", "q0"])
    with pytest.raises(ValueError):
        _spec({"h1": ["m/a"]}, ring=[("hk", "c1"), ("hk", "c2")])


# --- votes and items -----------------------------------------------------------


def _votes_for(spec, label, cost=1_000, failed=lambda hk, cell: False):
    ck = dict(spec.ring)
    out = []
    for cell in spec.cells:
        for hk in spec.assignment[cell]:
            out.append(Vote(spec.epoch, cell[0], cell[1], hk, ck[hk], label(hk, cell), cost,
                            failed(hk, cell)))
    return out


def _cost_map(**kw):
    m = {e: 1_000 for e in MENU}
    m.update(kw)
    return m


def test_a_right_answer_against_a_wrong_one_is_100_points_and_equal_picks_count_as_zero():
    s = _spec({"x": ["m/a", "m/c"], "y": ["m/b", "m/c"]}, baseline="")
    votes = _votes_for(s, lambda hk, c: Y if c[1] == "m/a" else 0)
    items = exam_v5.comparison_items([s], votes, "x", "y", frozenset(), _cost_map(x=1000, y=1000),
                                     lambda q: "all")
    assert items == {"all": [100 * arena_score.UNITS_PER_POINT, 0]}


def test_both_compared_operators_votes_never_count():
    s = _spec({"x": ["m/a"], "y": ["m/b"]}, baseline="")
    ck = dict(s.ring)
    bad = {ck[s.assignment[("q0", "m/a")][0]]}
    # the excluded operator lies; the others agree x is right
    votes = _votes_for(s, lambda hk, c: (0 if ck[hk] in bad else Y) if c[1] == "m/a" else 0)
    got = exam_v5.comparison_items([s], votes, "x", "y", frozenset(bad), _cost_map(x=1, y=1),
                                   lambda q: "s")
    assert got == {"s": [100 * arena_score.UNITS_PER_POINT]}
    # excluding every voter of a cell loses the question for this comparison only
    every = frozenset(ck[h] for h in s.assignment[("q0", "m/a")])
    assert exam_v5.comparison_items([s], votes, "x", "y", every, _cost_map(x=1, y=1),
                                    lambda q: "s") == {}


def test_only_assigned_hotkeys_vote_and_a_double_vote_is_dropped():
    s = _spec({"x": ["m/a"], "y": ["m/b"]}, baseline="")
    honest = _votes_for(s, lambda hk, c: Y if c[1] == "m/a" else 0)
    stranger = [Vote(s.epoch, "q0", "m/a", "hk-stranger", "ck-x", 0, 5, False)] * 5
    a1 = s.assignment[("q0", "m/a")][0]
    double = [Vote(s.epoch, "q0", "m/a", a1, dict(s.ring)[a1], 0, 1_000, False)]
    got = exam_v5.comparison_items([s], honest + stranger + double, "x", "y", frozenset(),
                                   _cost_map(x=1, y=1), lambda q: "s")
    assert got == {"s": [100 * arena_score.UNITS_PER_POINT]}


def test_failed_votes_drop_and_an_all_failed_cell_is_a_free_wrong_answer():
    s = _spec({"x": ["m/a"], "y": ["m/b"]}, baseline="")
    votes = _votes_for(s, lambda hk, c: Y, failed=lambda hk, c: c[1] == "m/a")
    got = exam_v5.comparison_items([s], votes, "x", "y", frozenset(), _cost_map(x=1, y=1),
                                   lambda q: "s")
    v = arena_score.value_scale_nd(1, 1)
    assert got == {"s": [arena_score.item_units(0, Y, 0, 1_000, v)]}
    one = [v for v in votes if not v.failed] + [next(v for v in votes if v.failed)]
    assert exam_v5.comparison_items([s], one, "x", "y", frozenset(), _cost_map(x=1, y=1),
                                    lambda q: "s") == {}


def test_median_label_and_cost_of_the_votes():
    s = _spec({"x": ["m/a"], "y": ["m/b"]}, baseline="")
    ys = iter([Y, Y, 0])
    votes = []
    ck = dict(s.ring)
    for cell in s.cells:
        for i, hk in enumerate(s.assignment[cell]):
            if cell[1] == "m/a":
                votes.append(Vote(s.epoch, *cell, hk, ck[hk], next(ys), [10, 20, 90][i], False))
            else:
                votes.append(Vote(s.epoch, *cell, hk, ck[hk], 0, 0, False))
    got = exam_v5.comparison_items([s], votes, "x", "y", frozenset(), _cost_map(x=5, y=5),
                                   lambda q: "s")
    v = arena_score.value_scale_nd(5, 5)
    assert got == {"s": [arena_score.item_units(Y, 0, 20, 0, v)]}


def test_v_is_set_from_the_heads_label_free_mean_costs():
    s = _spec({"x": ["m/a", "m/b"], "y": ["m/b", "m/c"]}, baseline="")
    costs = {"m/a": 100, "m/b": 300, "m/c": 500, "m/d": 1}
    assert exam_v5.mean_pick_cost_nd([s], "x", costs) == 200
    assert exam_v5.mean_pick_cost_nd([s], "y", costs) == 400
    assert exam_v5.mean_pick_cost_nd([s], "nobody", costs) == 0


def test_baseline_items_compare_x_with_b_star_on_the_baseline_questions():
    s = _spec({"x": ["m/b", "m/a"], "y": ["m/c", "m/a"]}, baseline="m/a")
    votes = _votes_for(s, lambda hk, c: Y if c[1] == "m/a" else 0)
    got = exam_v5.baseline_items([s], votes, "x", frozenset(), _cost_map(x=1000), lambda q: "s")
    assert got == {"s": [-100 * arena_score.UNITS_PER_POINT, 0]}
    assert exam_v5.baseline_items([s], votes, "absent", frozenset(), _cost_map(), lambda q: "s") == {}


def test_items_are_independent_of_vote_and_spec_order():
    rng = random.Random(9)
    specs = []
    for e in range(3):
        picks = {h: [rng.choice(MENU) for _ in range(60)] for h in ("x", "y", "z")}
        specs.append(build_spec(e, bytes([e]) * 32, picks, MENU, [f"e{e}q{i}" for i in range(60)],
                                RING, "m/a"))
    votes = [v for s in specs for v in _votes_for(
        s, lambda hk, c: rng.choice([0, Y, Y // 2]), cost=rng.randint(1, 10**6),
        failed=lambda hk, c: rng.random() < 0.1)]
    costs = _cost_map(x=700, y=900)
    a = exam_v5.comparison_items(specs, votes, "x", "y", frozenset({"ck1"}), costs,
                                 lambda q: q[:2])
    b = exam_v5.comparison_items(list(reversed(specs)), list(reversed(votes)), "x", "y",
                                 frozenset({"ck1"}), costs, lambda q: q[:2])
    assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}
