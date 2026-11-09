"""Protocol v2 champion scoring: paired NV, the seat, ghosts.

docs/INVARIANTS.md, "I3 — one champion; taking the seat needs the margin,
holding it needs none".
"""
import math
import random

import numpy as np
import pytest

from fugal_subnet.champion import (
    NV_UNITS_PER_POINT,
    Candidate,
    EpochSlice,
    HeadEpoch,
    SeatParams,
    SeatState,
    Window,
    pair_seed,
    pair_units,
    paired_test,
    question_nv_units,
    seat_step,
    window_accuracy,
)

Q = 300
M = 3                      # model 0 cheap/weak, 1 mid, 2 expensive/strong
CATS = ["knowledge"] * 90 + ["math"] * 75 + ["hard"] * 90 + ["instruction"] * 45
SPEND = np.array([0.0001, 0.001, 0.01])          # $ per question, per model
ACC = np.array([0.55, 0.70, 0.85])                # what each model gets right

PARAMS = SeatParams(
    margin_points=0.2, confidence=0.90, resamples=400, accuracy_value=30.0,
    hard_weight=2.0, hard_gate_min_questions=100,
    ghost_off=3, ghost_retire=24, wilson_confidence=0.95, min_accuracy=0.30,
)


def make_slice(epoch, seed=0):
    rng = np.random.default_rng(seed + epoch)
    # Model k solves question q when q's difficulty is below its accuracy, so
    # stronger models solve a superset: a realistic, router-friendly world.
    difficulty = rng.random(Q)
    grades = (difficulty[:, None] < ACC[None, :]).astype(np.float64)
    return EpochSlice(
        epoch=epoch,
        question_ids=tuple(f"e{epoch}q{i}" for i in range(Q)),
        categories=tuple(CATS),
        hard=tuple(c == "hard" for c in CATS),
        grades=grades,
        spends=np.tile(SPEND, (Q, 1)),
    ), difficulty


def oracle_router(difficulty):
    """Cheapest model that solves each question; the strong model when none does."""
    routed = np.full(Q, 2)
    for k in (1, 0):
        routed = np.where(difficulty < ACC[k], k, routed)
    return routed


def constant(k):
    return lambda difficulty: np.full(Q, k)


def noisy(policy, flip, seed):
    """A policy that, on a share of questions, routes to the strong model instead."""
    def run(difficulty):
        r = policy(difficulty).copy()
        rng = np.random.default_rng(seed + int(difficulty[0] * 1e9))
        r[rng.random(Q) < flip] = 2
        return r
    return run


def build_window(policies, epochs=range(10), size=48):
    w = Window.empty(size)
    for e in epochs:
        sl, diff = make_slice(e)
        answers = {
            h: HeadEpoch(routed=np.asarray(p(diff), dtype=np.int64), failed=np.zeros(Q, bool))
            for h, p in policies.items()
        }
        w = w.add_epoch(sl, answers)
    return w


def cand(h, seniority=0, eligible=True, since=0, burned_in=True):
    return Candidate(head_hash=h, seniority=(seniority,), eligible=eligible,
                     since_epoch=since, burned_in=burned_in)


def units(w, a, b):
    return pair_units(w, a, b, accuracy_value=30.0, hard_weight=2.0)


def seated(h, hw=0.0):
    return SeatState(champion=h, champion_since=0, high_water=hw)


# --- NV and the paired test -------------------------------------------------


def test_nv_is_zero_for_identical_heads_and_antisymmetric_without_cost():
    w = build_window({"a": oracle_router, "b": noisy(oracle_router, 0.3, 1),
                      "c": oracle_router})
    ac = units(w, "a", "c")
    assert all(not x.any() for x in ac.by_category.values())
    assert not paired_test(ac.by_category, margin_points=0.2, confidence=0.9,
                           resamples=200, seed=1).takes_seat
    # V is the bar's own spend, so it differs by direction; with the cost term
    # off NV is exactly antisymmetric.
    one = pair_units(w, "a", "b", accuracy_value=1e300, hard_weight=2.0)
    two = pair_units(w, "b", "a", accuracy_value=1e300, hard_weight=2.0)
    for c in one.by_category:
        assert np.array_equal(one.by_category[c], -two.by_category[c])


def test_v_is_the_champions_own_mean_spend_on_the_paired_questions():
    w = build_window({"a": oracle_router, "champ": constant(1)})
    pu = units(w, "a", "champ")
    assert pu.value_per_correct == pytest.approx(30.0 * SPEND[1])
    assert pu.n_questions == 10 * Q


def test_a_bar_that_spent_nothing_drops_the_cost_term():
    w = build_window({"a": oracle_router, "free": constant(0)})
    for sl in w.slices.values():
        sl.spends[:] = 0.0
    assert units(w, "a", "free").value_per_correct is None


def test_question_nv_matches_the_definition():
    # One extra correct answer, at 0.01 $ more, V = 0.3: 100*(1 - 0.01/0.3) points.
    u = question_nv_units(1.0, 0.0, 0.011, 0.001, hard=False,
                          value_per_correct=0.3, hard_weight=2.0)
    assert u == round(100 * (1 - 0.01 / 0.3) * NV_UNITS_PER_POINT)
    hard = question_nv_units(1.0, 0.0, 0.0, 0.0, hard=True, value_per_correct=0.3, hard_weight=2.0)
    assert hard == 200 * NV_UNITS_PER_POINT


def test_the_paired_test_is_exact_and_order_independent():
    rng = random.Random(3)
    u = {c: [rng.randrange(-10**8, 10**8) for _ in range(500)] for c in ("a", "b", "c")}
    u["a"] = [x + 10**6 for x in u["a"]]
    r1 = paired_test(u, margin_points=0.0, confidence=0.9, resamples=300, seed=99)
    r2 = paired_test(dict(reversed(list(u.items()))), margin_points=0.0,
                     confidence=0.9, resamples=300, seed=99)
    assert r1 == r2


def test_the_point_mean_is_a_necessary_condition():
    # The rule is "mean > margin AND LCB > margin". A mean at or below the
    # margin never takes the seat and no bootstrap runs; the reported LCB is
    # then the mean, so LCB <= mean holds by construction on this path.
    margin_units = round(0.2 * NV_UNITS_PER_POINT)
    res = paired_test({"x": [margin_units] * 100}, margin_points=0.2, confidence=0.9,
                      resamples=50, seed=0)
    assert not res.takes_seat and res.lcb_units == res.mean_units == margin_units


def test_the_lower_bound_does_not_exceed_the_mean_on_real_data():
    w = build_window({"a": oracle_router, "b": constant(1)})
    res = paired_test(units(w, "a", "b").by_category, margin_points=0.2, confidence=0.9,
                      resamples=400, seed=5)
    assert res.takes_seat and res.lcb_units <= res.mean_units


def test_a_challenger_just_below_the_margin_does_not_take_and_just_above_does():
    n = 3000
    for delta, expected in ((-1, False), (+1, True)):
        vals = [round(0.2 * NV_UNITS_PER_POINT) + 50 * delta + (i % 3 - 1) for i in range(n)]
        res = paired_test({"x": vals}, margin_points=0.2, confidence=0.9, resamples=200, seed=7)
        assert res.takes_seat is expected, (delta, res)


def test_the_seed_depends_on_epoch_and_both_heads():
    seeds = {pair_seed(e, a, b) for e in ("e1", "e2") for a in ("x", "y") for b in ("x", "y")}
    assert len(seeds) == 8


# --- the first champion -------------------------------------------------------


def test_nothing_eligible_means_an_empty_seat():
    w = build_window({"a": oracle_router})
    d = seat_step(SeatState(), 9, "e9", w, {"a": cand("a", eligible=False)}, params=PARAMS)
    assert d.champion is None and d.event == "empty"


def test_the_most_senior_eligible_head_takes_the_empty_seat_and_sets_the_mark():
    w = build_window({"junior": oracle_router, "senior": constant(2)})
    d = seat_step(SeatState(), 9, "e9", w,
                  {"junior": cand("junior", 5), "senior": cand("senior", 1)}, params=PARAMS)
    assert d.event == "first" and d.champion == "senior"
    acc, _ = window_accuracy(w, "senior")
    assert 0.0 < d.state.high_water <= acc


def test_a_constant_policy_can_hold_the_seat_until_a_router_beats_it_by_the_margin():
    # Accepted (INVARIANTS): with no virtual champion a constant head can be
    # first. Any router that beats it by the margin unseats it.
    w = build_window({"const": constant(2), "router": oracle_router})
    first = seat_step(SeatState(), 9, "e9", w, {"const": cand("const", 0)}, params=PARAMS)
    assert first.champion == "const"
    taken = seat_step(first.state, 9, "e9", w,
                      {"const": cand("const", 0), "router": cand("router", 1)}, params=PARAMS)
    assert taken.event == "take" and taken.champion == "router"
    assert taken.state.history[0] == "const"


# --- hold, take, hysteresis -----------------------------------------------------


def test_the_champion_holds_against_an_equal_or_slightly_better_challenger():
    w = build_window({"champ": oracle_router, "copy": oracle_router,
                      "nudge": noisy(oracle_router, 0.002, 11)})
    for challenger in ("copy", "nudge"):
        d = seat_step(seated("champ"), 9, "e9", w,
                      {"champ": cand("champ", 0), challenger: cand(challenger, 1)}, params=PARAMS)
        assert d.event == "hold" and d.champion == "champ"


def test_k_equal_juniors_earn_nothing():
    juniors = {f"j{i}": oracle_router for i in range(8)}
    w = build_window({"champ": oracle_router, **juniors})
    cands = {"champ": cand("champ", 0), **{h: cand(h, i + 1) for i, h in enumerate(juniors)}}
    d = seat_step(seated("champ"), 9, "e9", w, cands, params=PARAMS)
    assert d.champion == "champ" and d.event == "hold"


def test_a_better_challenger_takes_the_seat_from_a_seated_champion():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1)}, params=PARAMS)
    assert d.event == "take" and d.champion == "better"
    assert d.state.history[0] == "champ"


def test_an_ineligible_challenger_never_takes():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1, eligible=False)},
                  params=PARAMS)
    assert d.event == "hold"


def test_burn_in_is_needed_to_take_a_seat_but_not_to_hold_or_resume_one():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    # A better head that has not completed burn-in cannot take the seat ...
    d = seat_step(seated("champ"), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1, burned_in=False)},
                  params=PARAMS)
    assert d.event == "hold"
    # ... nor an empty one.
    e = seat_step(SeatState(), 9, "e9", w, {"better": cand("better", 1, burned_in=False)},
                  params=PARAMS)
    assert e.event == "empty"
    # A champion whose burn-in restarted after a miss still holds ...
    h = seat_step(seated("champ"), 9, "e9", w, {"champ": cand("champ", 0, burned_in=False)},
                  params=PARAMS)
    assert h.event == "hold"
    # ... and resumes inside its ghost window.
    lost = seat_step(seated("champ"), 9, "e9", w, {}, params=PARAMS)
    back = seat_step(lost.state, 10, "e10", w, {"champ": cand("champ", 0, burned_in=False)},
                     params=PARAMS)
    assert back.event == "hold" and back.champion == "champ"


def broken(difficulty):
    return np.full(Q, 0)


def failing_window(policies, failing):
    """Like build_window, but every call of the heads in `failing` fails."""
    w = Window.empty(48)
    for e in range(10):
        sl, diff = make_slice(e)
        w = w.add_epoch(sl, {
            h: HeadEpoch(np.asarray(p(diff), dtype=np.int64), np.full(Q, h in failing))
            for h, p in policies.items()
        })
    return w


def test_a_broken_head_cannot_take_an_empty_seat():
    # Every call failing: accuracy 0, under the 0.30 floor. The junior working
    # router is first instead, though the broken head is senior.
    w = failing_window({"broken": broken, "router": oracle_router}, failing={"broken"})
    d = seat_step(SeatState(), 9, "e9", w,
                  {"broken": cand("broken", 0), "router": cand("router", 1)}, params=PARAMS)
    assert d.event == "first" and d.champion == "router"


def test_a_champion_below_the_floor_loses_the_seat():
    w = failing_window({"champ": oracle_router}, failing={"champ"})
    d = seat_step(seated("champ"), 9, "e9", w, {"champ": cand("champ", 0)}, params=PARAMS)
    assert d.event == "ghost" and d.champion is None
    # ... and does not resume while it stays below it.
    again = seat_step(d.state, 10, "e10", w, {"champ": cand("champ", 0)}, params=PARAMS)
    assert again.champion is None


def test_the_high_water_mark_gates_and_only_rises():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    acc, _ = window_accuracy(w, "better")
    blocked = seat_step(seated("champ", hw=acc + 0.01), 9, "e9", w,
                        {"champ": cand("champ", 0), "better": cand("better", 1)}, params=PARAMS)
    assert blocked.event == "hold"
    taken = seat_step(seated("champ", hw=0.5), 9, "e9", w,
                      {"champ": cand("champ", 0), "better": cand("better", 1)}, params=PARAMS)
    assert taken.champion == "better" and 0.5 < taken.state.high_water <= acc
    held = seat_step(seated("champ", hw=0.999), 9, "e9", w, {"champ": cand("champ", 0)},
                     params=PARAMS)
    assert held.state.high_water == 0.999   # holding never lowers it


# --- ghosts ------------------------------------------------------------------------


def test_a_missing_champion_leaves_a_ghost_bar_then_the_next_head_is_first():
    w = build_window({"champ": oracle_router, "copy": oracle_router}, epochs=range(10))
    s = seated("champ")
    events = []
    for e in range(10, 15):
        sl, diff = make_slice(e)
        w = w.add_epoch(sl, {"copy": HeadEpoch(np.asarray(oracle_router(diff)), np.zeros(Q, bool))})
        d = seat_step(s, e, f"e{e}", w, {"copy": cand("copy", 1)}, params=PARAMS)
        events.append((d.event, d.champion))
        s = d.state
    # Ghost for GHOST_OFF = 3 epochs (10, 11, 12): the share burns, and an
    # equal copy cannot beat the ghost. Then the seat is empty and the copy,
    # the only eligible head, is first.
    assert events[:3] == [("ghost", None)] * 3
    assert events[3] == ("first", "copy")


def test_the_marks_reset_when_the_last_ghost_expires_with_the_seat_empty():
    # A departed champion's bar must not burn the share indefinitely (no-burn
    # preference): once its ghost expires, a weaker head can be first.
    w = build_window({"champ": oracle_router, "weak": constant(0)}, epochs=range(10))
    s = seated("champ", hw=0.99)
    for e in range(10, 13):          # ghost epochs 10, 11, 12
        sl, _ = make_slice(e)
        w = w.add_epoch(sl, {"weak": HeadEpoch(np.zeros(Q, np.int64), np.zeros(Q, bool))})
        s = seat_step(s, e, f"e{e}", w, {"weak": cand("weak", 1)}, params=PARAMS).state
        assert s.high_water == 0.99
    sl, _ = make_slice(13)
    w = w.add_epoch(sl, {"weak": HeadEpoch(np.zeros(Q, np.int64), np.zeros(Q, bool))})
    d = seat_step(s, 13, "e13", w, {"weak": cand("weak", 1)}, params=PARAMS)
    assert d.event == "first" and d.champion == "weak"
    acc, _ = window_accuracy(w, "weak")
    assert d.state.high_water <= acc < 0.99


def test_a_challenger_that_beats_the_ghost_takes_the_seat_inside_the_window():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w, {"better": cand("better", 1)}, params=PARAMS)
    assert d.event == "take" and d.champion == "better" and d.state.ghosts == ()


def test_a_champion_that_returns_inside_its_ghost_window_resumes():
    w = build_window({"champ": oracle_router, "copy": oracle_router})
    lost = seat_step(seated("champ"), 9, "e9", w, {"copy": cand("copy", 1)}, params=PARAMS)
    assert lost.event == "ghost"
    back = seat_step(lost.state, 10, "e10", w,
                     {"champ": cand("champ", 0), "copy": cand("copy", 1)}, params=PARAMS)
    assert back.event == "hold" and back.champion == "champ"
    assert back.state.ghosts == () and "champ" not in back.state.history


def test_a_retired_champion_bars_for_the_retire_window_and_does_not_resume():
    w = build_window({"champ": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w, {}, retired={"champ"}, params=PARAMS)
    (g,) = d.state.ghosts
    assert g.reason == "retired" and g.last_epoch == 9 + PARAMS.ghost_retire - 1
    assert d.event == "ghost" and d.champion is None
    back = seat_step(d.state, 10, "e10", w, {"champ": cand("champ")}, params=PARAMS)
    assert back.champion is None


# --- determinism and the window -------------------------------------------------


def test_the_seat_does_not_depend_on_candidate_order():
    pols = {"champ": noisy(oracle_router, 0.6, 5), "a": oracle_router,
            "b": noisy(oracle_router, 0.1, 9), "c": constant(1)}
    w = build_window(pols)
    cands = [cand(h, i) for i, h in enumerate(pols)]
    first = seat_step(seated("champ"), 9, "e9", w, {c.head_hash: c for c in cands}, params=PARAMS)
    for i in range(3):
        random.Random(i).shuffle(cands)
        again = seat_step(seated("champ"), 9, "e9", w, {c.head_hash: c for c in cands},
                          params=PARAMS)
        assert again == first


def test_the_window_keeps_only_the_last_size_epochs():
    w = build_window({"a": oracle_router}, epochs=range(10), size=4)
    assert sorted(w.slices) == [6, 7, 8, 9]
    assert w.epochs_of("a") == [6, 7, 8, 9]


def test_re_adding_an_epoch_replaces_it():
    # Restart idempotence: a retried epoch (crash between the window and the
    # JSON state) or a --once re-run re-adds the same epoch; the window must
    # come out the same, not fail.
    w = build_window({"a": oracle_router, "b": constant(1)}, epochs=range(10))
    sl, diff = make_slice(9)
    again = w.add_epoch(sl, {"a": HeadEpoch(np.asarray(oracle_router(diff)), np.zeros(Q, bool)),
                             "b": HeadEpoch(np.full(Q, 1), np.zeros(Q, bool))})
    assert sorted(again.slices) == sorted(w.slices)
    for h in ("a", "b"):
        assert np.array_equal(again.heads[h][9].routed, w.heads[h][9].routed)
    # A head absent from the replacement is absent from that epoch.
    only_a = w.add_epoch(sl, {"a": HeadEpoch(np.asarray(oracle_router(diff)), np.zeros(Q, bool))})
    assert 9 not in only_a.heads["b"] and 8 in only_a.heads["b"]


def test_a_window_that_is_not_trusted_makes_no_take():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1)},
                  params=PARAMS, allow_take=False)
    assert d.event == "hold"
    e = seat_step(SeatState(), 9, "e9", w, {"better": cand("better", 1)}, params=PARAMS,
                  allow_take=False)
    assert e.event == "empty"


def test_a_failed_call_grades_zero():
    sl, _ = make_slice(0)
    failed = np.zeros(Q, bool)
    failed[:10] = True
    w = Window.empty(4).add_epoch(sl, {"a": HeadEpoch(np.full(Q, 2), failed)})
    g, _ = w.head_vectors("a", 0)
    assert not g[:10].any()


def test_window_accuracy_is_exactly_summed():
    w = build_window({"a": oracle_router})
    acc, n = window_accuracy(w, "a")
    flat = [float(x) for e in w.epochs_of("a") for x in w.head_vectors("a", e)[0]]
    assert n == len(flat) and acc == math.fsum(flat) / len(flat)


# --- review fixes ------------------------------------------------------------------


def run_window(schedule, policies, size=48):
    """A window over len(schedule) epochs; schedule[e] is the set of heads scored in e."""
    w = Window.empty(size)
    for e, present in enumerate(schedule):
        sl, diff = make_slice(e)
        w = w.add_epoch(sl, {
            h: HeadEpoch(np.asarray(policies[h](diff), dtype=np.int64), np.zeros(Q, bool))
            for h in present
        })
    return w


def test_a_withheld_epoch_never_enters_the_pairing_and_a_miss_restarts_it():
    pols = {"champ": oracle_router, "c": oracle_router}
    both = {"champ", "c"}
    # c serves 0-3, sits out 4, serves 5-9.
    sched = [both] * 4 + [{"champ"}] + [both] * 5
    w = run_window(sched, pols)
    assert w.current_run("c") == [5, 6, 7, 8, 9]
    assert units(w, "c", "champ").n_questions == 5 * Q     # epochs 0-3 are gone
    # A champion's own gaps do not break the challenger's run.
    sched2 = [both] * 4 + [{"c"}] + [both] * 5
    assert run_window(sched2, pols).current_run("c") == list(range(10))
    assert units(run_window(sched2, pols), "c", "champ").n_questions == 9 * Q


def test_the_take_gates_read_only_the_current_run():
    # c was terrible before its miss and is good after it; only "after" counts.
    pols = {"champ": noisy(oracle_router, 0.6, 5), "c": constant(0), "c2": oracle_router}
    w = Window.empty(48)
    for e in range(10):
        sl, diff = make_slice(e)
        ans = {"champ": HeadEpoch(np.asarray(pols["champ"](diff)), np.zeros(Q, bool))}
        if e < 4:
            ans["c"] = HeadEpoch(np.asarray(pols["c"](diff)), np.zeros(Q, bool))
        elif e > 4:
            ans["c"] = HeadEpoch(np.asarray(pols["c2"](diff)), np.zeros(Q, bool))
        w = w.add_epoch(sl, ans)
    full, _ = window_accuracy(w, "c")
    run, n = window_accuracy(w, "c", run_only=True)
    assert n == 5 * Q and run > full


def test_a_missed_ghost_whose_owner_commits_a_new_head_becomes_a_retire_ghost():
    w = build_window({"champ": oracle_router})
    lost = seat_step(seated("champ"), 9, "e9", w, {}, params=PARAMS)
    (g,) = lost.state.ghosts
    assert g.reason == "missed" and g.last_epoch == 9 + PARAMS.ghost_off - 1
    up = seat_step(lost.state, 10, "e10", w, {}, retired={"champ"}, params=PARAMS)
    (g2,) = up.state.ghosts
    assert g2.reason == "retired" and g2.last_epoch == 9 + PARAMS.ghost_retire - 1
    # ... and it no longer resumes.
    back = seat_step(up.state, 11, "e11", w, {"champ": cand("champ")}, params=PARAMS)
    assert back.champion is None


def test_a_take_needs_the_minimum_paired_questions():
    import dataclasses

    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    strict = dataclasses.replace(PARAMS, min_paired_questions=10 * Q + 1)
    d = seat_step(seated("champ"), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1)}, params=strict)
    assert d.event == "hold"
    ok = dataclasses.replace(PARAMS, min_paired_questions=10 * Q)
    assert seat_step(seated("champ"), 9, "e9", w,
                     {"champ": cand("champ", 0), "better": cand("better", 1)},
                     params=ok).event == "take"


def test_shipped_min_paired_questions_is_1500():
    from fugal_subnet.config import CHAMPION_MIN_PAIRED_QUESTIONS
    assert CHAMPION_MIN_PAIRED_QUESTIONS == 1500


def test_can_hold_fails_closed_with_no_window_records():
    w = build_window({"other": oracle_router})
    d = seat_step(seated("champ"), 9, "e9", w, {"champ": cand("champ")}, params=PARAMS)
    assert d.champion is None and d.event == "ghost"


def test_out_of_range_routed_indices_are_refused():
    sl, _ = make_slice(0)
    for bad in (-1, M):
        r = np.zeros(Q, np.int64)
        r[3] = bad
        with pytest.raises(ValueError, match="outside"):
            Window.empty(4).add_epoch(sl, {"a": HeadEpoch(r, np.zeros(Q, bool))})


def test_history_is_bounded():
    from fugal_subnet.champion import HISTORY_KEEP
    s = SeatState(history=tuple(f"h{i}" for i in range(50)))
    assert len(s.history) == HISTORY_KEEP and s.history[0] == "h0"


def test_per_question_units_are_clipped():
    from fugal_subnet.champion import UNIT_CLIP
    u = question_nv_units(0.0, 0.0, 1e300, 0.0, hard=False, value_per_correct=1e-300,
                          hard_weight=2.0)
    assert u == -UNIT_CLIP


def test_the_lower_bound_index_rounds():
    # (1 - 0.9) * 2000 is 199.99999999999997; the bound is the 200th order
    # statistic (0-based), not the 199th.
    assert round((1 - 0.9) * 2000) == 200 and int((1 - 0.9) * 2000) == 199
    import inspect

    from fugal_subnet import champion
    assert "k = min(round(" in inspect.getsource(champion.paired_test)


# --- the pinned anchor (V independent of the champion) ---------------------------


def _cheap_vs_accurate_window(epochs=10):
    """Champion: always the cheap model ($0.0004/q). Challenger: 5 points more
    accurate, at $0.01/q (25x the champion's spend)."""
    rng = np.random.default_rng(42)
    w = Window.empty(48)
    for e in range(epochs):
        cheap = (rng.random(Q) < 0.60).astype(np.float64)
        # The dear model solves everything the cheap one does plus 5% more.
        dear = np.maximum(cheap, (rng.random(Q) < 0.125).astype(np.float64))
        grades = np.stack([cheap, dear], axis=1)
        spends = np.tile(np.array([0.0004, 0.01]), (Q, 1))
        sl = EpochSlice(epoch=e, question_ids=tuple(f"e{e}q{i}" for i in range(Q)),
                        categories=tuple(CATS), hard=tuple(c == "hard" for c in CATS),
                        grades=grades, spends=spends)
        w = w.add_epoch(sl, {"champ": HeadEpoch(np.zeros(Q, np.int64), np.zeros(Q, bool)),
                             "router": HeadEpoch(np.ones(Q, np.int64), np.zeros(Q, bool))})
    return w


def test_a_cheap_champion_no_longer_blocks_a_more_accurate_router():
    import dataclasses

    w = _cheap_vs_accurate_window()
    acc_c, _ = window_accuracy(w, "champ")
    acc_r, _ = window_accuracy(w, "router")
    assert 0.04 < acc_r - acc_c < 0.07                    # about +5 points
    cands = {"champ": cand("champ", 0), "router": cand("router", 1)}
    # Champion-anchored V (= 30 x $0.0004): the router's spend costs ~80 points.
    legacy = seat_step(seated("champ"), 9, "e9", w, cands, params=PARAMS)
    assert legacy.event == "hold"
    # Pinned anchor (~$0.026/q, the v2 Fable expectation): the spend costs ~1.2.
    anchored = dataclasses.replace(PARAMS, anchor_spend=0.0258)
    taken = seat_step(seated("champ"), 9, "e9", w, cands, params=anchored)
    assert taken.event == "take" and taken.champion == "router"


def test_the_anchor_sets_v_whoever_holds_the_seat():
    w = _cheap_vs_accurate_window(3)
    a = pair_units(w, "router", "champ", accuracy_value=30.0, hard_weight=2.0,
                   anchor_spend=0.0258)
    b = pair_units(w, "champ", "router", accuracy_value=30.0, hard_weight=2.0,
                   anchor_spend=0.0258)
    assert a.value_per_correct == b.value_per_correct == pytest.approx(30 * 0.0258)
    for c in a.by_category:                                # exactly antisymmetric now
        assert np.array_equal(a.by_category[c], -b.by_category[c])
    with pytest.raises(ValueError):
        pair_units(w, "router", "champ", accuracy_value=30.0, hard_weight=2.0, anchor_spend=0.0)


def test_the_shipped_anchor_is_fable_priced_from_pinned_files():
    from fugal_subnet.config import CHAMPION_ANCHOR_MODEL
    from fugal_subnet.v2_epoch import anchor_spend_per_question, seat_params_from_config

    assert CHAMPION_ANCHOR_MODEL == "anthropic/claude-fable-5.1"
    spend = anchor_spend_per_question(CHAMPION_ANCHOR_MODEL)
    assert 0.02 < spend < 0.03
    assert seat_params_from_config().anchor_spend == spend
    with pytest.raises(KeyError):
        anchor_spend_per_question("no/such-model")


def test_a_lower_bound_exactly_at_the_margin_does_not_take_the_seat():
    """'Exceed the margin' is strict. Nine questions at the margin and one far
    above it: the mean clears the margin, but at 90% the lower bound is a
    resample without the outlier, exactly the margin. (A planted `>=` for `>`
    survived every test.)"""
    m = round(0.2 * NV_UNITS_PER_POINT)
    r = paired_test({"math": [m] * 9 + [11 * m]}, margin_points=0.2, confidence=0.9,
                    resamples=2000, seed=1)
    assert r.mean_units > m                 # the point estimate is not what stops it
    assert r.lcb_units == m                 # the bound sits exactly on the margin
    assert not r.takes_seat


# --- rule gaps a planted-bug review found (each test fails if its rule breaks) ---


def test_a_mean_exactly_at_the_margin_never_takes_whatever_the_bound_says():
    """The point estimate is a necessary condition: a mean exactly AT the
    margin does not take the seat and pays no bootstrap, even where the bound
    alone would clear the margin (a low confidence puts the 'lower' bound at
    the 90th percentile of the resamples)."""
    m = round(0.2 * NV_UNITS_PER_POINT)
    at = [m - 100_000, m + 100_000] * 500                 # mean exactly the margin
    above = [m - 100_000, m + 100_002] * 500              # mean 1 unit above it
    for conf in (0.9, 0.1):
        r = paired_test({"x": at}, margin_points=0.2, confidence=conf, resamples=200, seed=3)
        assert r.mean_units == m and not r.takes_seat
        assert r.lcb_units == r.mean_units                # no bootstrap ran
    # Control: one unit above the margin, the same low-confidence bound takes.
    r = paired_test({"x": above}, margin_points=0.2, confidence=0.1, resamples=200, seed=3)
    assert r.takes_seat


def test_the_hard_accuracy_gate_blocks_a_take():
    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    acc_h, n_h = window_accuracy(w, "better", hard_only=True)
    assert n_h >= PARAMS.hard_gate_min_questions and acc_h < 0.99
    cands = {"champ": cand("champ", 0), "better": cand("better", 1)}
    # Control: no hard mark, the better head takes.
    assert seat_step(seated("champ"), 9, "e9", w, cands, params=PARAMS).event == "take"
    # A hard mark just above its hard accuracy blocks it, though its overall
    # accuracy clears the (zero) overall mark.
    st = SeatState(champion="champ", champion_since=0, high_water_hard=acc_h + 0.001)
    d = seat_step(st, 9, "e9", w, cands, params=PARAMS)
    assert d.event == "hold" and d.champion == "champ"
    assert any("hard accuracy" in msg for h, msg in d.detail if h == "better")


def test_a_new_champion_never_lowers_the_high_water_mark():
    """The marks never fall while a champion stands: a challenger that clears
    the mark but whose Wilson bound is below it leaves the mark where it was."""
    from fugal_subnet.champion import _wilson_lower

    w = build_window({"champ": noisy(oracle_router, 0.6, 5), "better": oracle_router})
    acc, n = window_accuracy(w, "better")
    hw = acc - 0.001
    assert _wilson_lower(acc, n, PARAMS.wilson_confidence) < hw <= acc
    d = seat_step(seated("champ", hw=hw), 9, "e9", w,
                  {"champ": cand("champ", 0), "better": cand("better", 1)}, params=PARAMS)
    assert d.event == "take" and d.champion == "better"
    assert d.state.high_water == hw


def test_holding_reads_the_whole_window_not_the_current_run():
    """The champion's side is every window epoch it was scored in. A champion
    that missed epoch 8 and failed every call in epoch 9 has a current run of
    one epoch at 0% accuracy, but its window accuracy is far above the floor:
    it holds. (A challenger's TAKE gates read only its run.)"""
    w = Window.empty(48)
    for e in range(10):
        sl, diff = make_slice(e)
        ans = {}
        if e != 8:
            ans["champ"] = HeadEpoch(np.asarray(oracle_router(diff), dtype=np.int64),
                                     np.full(Q, e == 9))
        w = w.add_epoch(sl, ans)
    run_acc, run_n = window_accuracy(w, "champ", run_only=True)
    acc, n = window_accuracy(w, "champ")
    assert run_n == Q and run_acc == 0.0 < PARAMS.min_accuracy <= acc and n == 9 * Q
    d = seat_step(seated("champ"), 9, "e9", w, {"champ": cand("champ", 0)}, params=PARAMS)
    assert d.event == "hold" and d.champion == "champ"
    # Control: failing every call in every epoch is below the floor and loses it.
    lost = failing_window({"champ": oracle_router}, failing={"champ"})
    assert seat_step(seated("champ"), 9, "e9", lost, {"champ": cand("champ", 0)},
                     params=PARAMS).event == "ghost"


def test_pair_units_weights_hard_questions_by_the_hard_weight():
    """Through the production path (pair_units -> _units), not the scalar
    question_nv_units: on a hard question a grade difference counts
    hard_weight times what it counts elsewhere."""
    w = build_window({"strong": constant(2), "weak": constant(0)})
    # A huge V drops the cost term to well under one unit.
    pu = pair_units(w, "strong", "weak", accuracy_value=1e300, hard_weight=2.0)
    epochs = w.epochs_of("strong")
    diff = np.concatenate([w.head_vectors("strong", e)[0] - w.head_vectors("weak", e)[0]
                           for e in epochs])
    cats = np.concatenate([np.asarray(w.slices[e].categories) for e in epochs])
    for c, got in pu.by_category.items():
        weight = 2.0 if c == "hard" else 1.0
        expect = np.rint(weight * 100.0 * NV_UNITS_PER_POINT * diff[cats == c]).astype(np.int64)
        assert np.array_equal(got, expect), c
    assert pu.by_category["hard"].max() == 200 * NV_UNITS_PER_POINT
    assert pu.by_category["math"].max() == 100 * NV_UNITS_PER_POINT
