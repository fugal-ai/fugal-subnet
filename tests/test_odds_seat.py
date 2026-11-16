"""Protocol v3: the exploration pool, the reference, and the seat decided on odds.

docs/INVARIANTS.md, "I3 — the seat goes to honest, sharp odds (protocol v3)"
and "I4 — every miner measures the models (protocol v3)".
"""
import dataclasses
import hashlib
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import odds_score
from fugal_subnet.champion import (
    EpochPairs,
    HeadEpoch,
    OddsParams,
    Pending,
    SeatState,
    Window,
    seat_step,
)
from fugal_subnet.explore_pool import (
    ExploreAnswer,
    Pair,
    capped,
    pairs_for_reveal,
    pool_pairs,
    private_implausible,
)
from fugal_subnet.head_odds import ODDS_SCALE
from tests.test_champion import PARAMS, Q, cand, constant, make_slice, oracle_router

NONCE = hashlib.sha256(b"odds").digest()
MODELS = ("m/cheap", "m/mid", "m/strong")
ACC = {"m/cheap": 0.55, "m/mid": 0.70, "m/strong": 0.85}
BENCH = ("knowledge", "math", "hard")          # one benchmark per category here
N_PAIRS = 300


def A(q, m, ck, correct=True, failed=False, private=True, ptok=10, ctok=100):
    return ExploreAnswer(q, m, ck, correct, failed, private, ptok, ctok)


# --- the pool ---------------------------------------------------------------------


def test_a_pair_is_the_coldkey_median_and_failed_calls_say_nothing():
    answers = [
        A("q1", "m/a", "ck1", True, private=False), A("q1", "m/a", "ck2", True, private=False),
        A("q1", "m/a", "ck3", False, private=False),
        A("q1", "m/a", "ck4", False, failed=True, private=False),       # dropped
        A("q2", "m/b", "ck1", False, ctok=300), A("q3", "m/b", "ck9", True, failed=True),
    ]
    pairs = pool_pairs(answers, nonce=NONCE, coldkey_cap=36)
    assert [(p.question_id, p.model, p.y, p.voters) for p in pairs] == [
        ("q1", "m/a", ODDS_SCALE, 3), ("q2", "m/b", 0, 1)]
    assert pairs[1].completion_tokens == 300 and pairs[0].prompt_tokens == 10


def test_one_operator_is_one_vote_however_many_hotkeys_it_runs():
    # ck1 answers the shared pair from five hotkeys, wrong each time; two others get it right.
    answers = [A("q", "m", "ck1", False, private=False) for _ in range(5)]
    answers += [A("q", "m", "ck2", True, private=False), A("q", "m", "ck3", True, private=False)]
    (pair,) = pool_pairs(answers, nonce=NONCE, coldkey_cap=36)
    assert pair.y == ODDS_SCALE and pair.voters == 3
    # An even split is a half; an operator's split answers are its mean.
    (pair,) = pool_pairs([A("q", "m", "ck1", True), A("q", "m", "ck2", False)],
                         nonce=NONCE, coldkey_cap=36)
    assert pair.y == ODDS_SCALE // 2
    (pair,) = pool_pairs([A("q", "m", "ck1", True), A("q", "m", "ck1", False),
                          A("q", "m", "ck1", True)], nonce=NONCE, coldkey_cap=36)
    assert pair.y == 6667 and pair.voters == 1


def test_an_operators_private_pairs_are_capped_and_it_cannot_choose_which():
    big = [A(f"q{i}", "m", "whale") for i in range(200)]
    small = [A(f"s{i}", "m", "minnow") for i in range(24)]
    shared = [A("shared", "m", ck, private=False) for ck in ("whale", "minnow")]
    kept = capped(big + small + shared, NONCE, 36)
    assert sum(a.coldkey == "whale" and a.private for a in kept) == 36
    assert sum(a.coldkey == "minnow" and a.private for a in kept) == 24
    assert sum(not a.private for a in kept) == 2            # shared answers are never cut
    # The same pairs whatever order the proofs arrived in.
    again = capped(list(reversed(big + small + shared)), NONCE, 36)
    assert {(a.coldkey, a.question_id) for a in kept} == {(a.coldkey, a.question_id) for a in again}
    # A different epoch keeps different ones: the operator cannot aim.
    other = capped(big, hashlib.sha256(b"other").digest(), 36)
    assert {a.question_id for a in other} != {a.question_id for a in kept if a.coldkey == "whale"}
    pairs = pool_pairs(big + small + shared, nonce=NONCE, coldkey_cap=36)
    assert len(pairs) == 36 + 24 + 1


def test_a_flagged_operator_contributes_nothing():
    answers = [A("q1", "m", "bad", False), A("q1", "m", "good", True), A("q2", "m", "bad", False)]
    pairs = pool_pairs(answers, nonce=NONCE, coldkey_cap=36, flagged={"bad"})
    assert [(p.question_id, p.y, p.voters) for p in pairs] == [("q1", ODDS_SCALE, 1)]


def test_implausible_private_answers_are_flagged_in_either_direction():
    expected = {(f"q{i}", "m"): 9000 for i in range(24)}
    honest = [A(f"q{i}", "m", "honest", i >= 2) for i in range(24)]            # 22 of 24
    zeros = [A(f"q{i}", "m", "poison", False) for i in range(24)]
    low = {(f"q{i}", "m"): 2000 for i in range(24)}
    flags = private_implausible(honest + zeros, expected, z=4, min_samples=8)
    assert set(flags) == {"poison"} and "below" in flags["poison"]
    assert set(private_implausible([A(f"q{i}", "m", "gold", True) for i in range(24)], low,
                                   z=4, min_samples=8)) == {"gold"}
    # Too few answers is never a flag; shared and failed answers are not judged.
    assert private_implausible(zeros[:7], expected, z=4, min_samples=8) == {}
    assert private_implausible([dataclasses.replace(a, private=False) for a in zeros],
                               expected, z=4, min_samples=8) == {}
    assert private_implausible([dataclasses.replace(a, failed=True) for a in zeros],
                               expected, z=4, min_samples=8) == {}


def test_a_near_certain_reference_does_not_flag_two_honest_misses():
    expected = {(f"q{i}", "m"): 9970 for i in range(24)}
    two_wrong = [A(f"q{i}", "m", "ck", i >= 2) for i in range(24)]
    assert private_implausible(two_wrong, expected, z=4, min_samples=8) == {}
    eight_wrong = [A(f"q{i}", "m", "ck", i >= 8) for i in range(24)]
    assert set(private_implausible(eight_wrong, expected, z=4, min_samples=8)) == {"ck"}


def test_the_pool_is_published_sorted_with_each_questions_benchmark():
    rows = pairs_for_reveal([Pair("q2", "m", 0, 1, 5, 7), Pair("q1", "m", ODDS_SCALE, 2, 5, 9)],
                            {"q1": "mmlu"})
    assert [r["question_id"] for r in rows] == ["q1", "q2"]
    assert rows[0] == {"question_id": "q1", "model": "m", "benchmark": "mmlu", "y": ODDS_SCALE,
                       "voters": 2, "prompt_tokens": 5, "completion_tokens": 9,
                       "weight": 1_000_000, "latency_ms": 0}
    assert rows[1]["benchmark"] == ""


# --- a small world: heads that predict, on pairs nobody chose ---------------------------


def epoch_pairs(epoch):
    """N_PAIRS random (question, model) pairs: a model is right when the
    question's difficulty is below its accuracy, so an oracle can know."""
    rng = np.random.default_rng(1000 + epoch)
    difficulty = rng.random(N_PAIRS)
    m_idx = rng.integers(0, len(MODELS), N_PAIRS)
    b_idx = rng.integers(0, len(BENCH), N_PAIRS)
    # Harder benchmarks: shift difficulty so base rates differ by benchmark.
    difficulty = np.clip(difficulty + 0.15 * (b_idx - 1), 0, 0.999)
    y = np.array([ODDS_SCALE if difficulty[i] < ACC[MODELS[m_idx[i]]] else 0
                  for i in range(N_PAIRS)], dtype=np.int32)
    pairs = EpochPairs(
        question_ids=tuple(f"x{epoch}_{i}" for i in range(N_PAIRS)),
        models=tuple(MODELS[i] for i in m_idx), benchmarks=tuple(BENCH[i] for i in b_idx),
        categories=tuple(BENCH[i] for i in b_idx), y=y,
        prompt_tokens=np.full(N_PAIRS, 10, dtype=np.int32),
        completion_tokens=np.full(N_PAIRS, 100, dtype=np.int32))
    return pairs, difficulty


def sharp(noise=0.0, seed=0):
    """Knows each question's difficulty up to `noise`: honest and sharp."""
    def predict(pairs, difficulty, epoch):
        rng = np.random.default_rng(seed + epoch)
        seen = difficulty + noise * rng.standard_normal(len(difficulty))
        acc = np.array([ACC[m] for m in pairs.models])
        # A logistic in the margin between the model's accuracy and the difficulty.
        p = 1 / (1 + np.exp(-(acc - seen) / max(noise, 0.02)))
        return np.rint(p * ODDS_SCALE).astype(np.int16)
    return predict


def lookup(pairs, difficulty, epoch):
    """A table of rates by model and benchmark: does not read the question."""
    table = {(m, b): float(np.clip(ACC[m] - 0.15 * (BENCH.index(b) - 1), 0.02, 0.98))
             for m in MODELS for b in BENCH}
    return np.array([round(table[(m, b)] * ODDS_SCALE)
                     for m, b in zip(pairs.models, pairs.benchmarks)], dtype=np.int16)


def gates(pairs, difficulty, epoch):
    """On/off switches: certain every time, right only as often as the base rate."""
    return np.array([ODDS_SCALE if m == "m/strong" else 0 for m in pairs.models], dtype=np.int16)


def only_strong(pairs, difficulty, epoch):
    """A head with a row for one model only."""
    out = sharp(0.05)(pairs, difficulty, epoch)
    return np.where(np.array([m == "m/strong" for m in pairs.models]), out, -1).astype(np.int16)


def world(heads, epochs=range(8), routing=None, size=48):
    """A window in which every head routes like `routing[h]` (the oracle by
    default, so the net-value guard is neutral) and predicts like `heads[h]`."""
    w = Window.empty(size)
    for e in epochs:
        present = {h: f for h, f in heads.items() if not isinstance(f, tuple) or e in f[1]}
        present = {h: (f[0] if isinstance(f, tuple) else f) for h, f in present.items()}
        sl, diff = make_slice(e)
        answers = {h: HeadEpoch(np.asarray((routing or {}).get(h, oracle_router)(diff), dtype=np.int64),
                                np.zeros(Q, bool)) for h in present}
        pairs, d = epoch_pairs(e)
        w = w.add_epoch(sl, answers, pairs, {h: f(pairs, d, e) for h, f in present.items()})
    return w


ODDS = dataclasses.replace(PARAMS, confidence=0.99, resamples=400, odds=OddsParams(
    margin_points=0.1, min_pairs=1500, nv_noninferiority=0.5, ref_prior=20, confirm_pairs=1500))


def ref_of(w):
    return odds_score.reference(w.pairs, 20)


# --- the reference ---------------------------------------------------------------------


def test_the_reference_never_sees_the_epoch_it_is_scored_on():
    one = world({"a": lookup}, epochs=[0])
    r = odds_score.reference(one.pairs, 20)
    # A single epoch: nothing else to learn from, so every pair gets the prior, 1/2.
    assert set(r.of(0).tolist()) == {ODDS_SCALE // 2}
    two = world({"a": lookup}, epochs=[0, 1])
    r2 = odds_score.reference(two.pairs, 20)
    # Epoch 0's reference is built from epoch 1 alone, and the reverse.
    ep1 = two.pairs[1]
    k = sum(int(y) for m, b, y in zip(ep1.models, ep1.benchmarks, ep1.y) if m == "m/mid")
    n = sum(1 for m in ep1.models if m == "m/mid")
    kc = sum(int(y) for m, b, y in zip(ep1.models, ep1.benchmarks, ep1.y)
             if (m, b) == ("m/mid", "math"))
    nc = sum(1 for m, b in zip(ep1.models, ep1.benchmarks) if (m, b) == ("m/mid", "math"))
    r_m = (Fraction(k, ODDS_SCALE) + 1) / (n + 2)
    want = (Fraction(kc, ODDS_SCALE) + 20 * r_m) / (nc + 20)
    i = next(i for i, (m, b) in enumerate(zip(two.pairs[0].models, two.pairs[0].benchmarks))
             if (m, b) == ("m/mid", "math"))
    assert int(r2.of(0)[i]) == int((2 * want * ODDS_SCALE + 1) // 2)


def test_the_reference_is_exact_and_does_not_depend_on_epoch_order():
    w = world({"a": lookup}, epochs=range(6))
    a = odds_score.reference(w.pairs, 20)
    b = odds_score.reference(dict(reversed(list(w.pairs.items()))), 20)
    assert all(np.array_equal(a.of(e), b.of(e)) for e in w.pairs)
    assert all(x.dtype == np.int32 and 0 <= x.min() and x.max() <= ODDS_SCALE for x in a.odds.values())


def test_reference_skill_gate():
    """A table of rates by model and benchmark has no skill against the
    reference; a head that reads the question has, and on/off switches have
    far less than none."""
    w = world({"sharp": sharp(0.05), "lookup": lookup, "gates": gates}, epochs=range(10))
    ref = ref_of(w)
    skill = {h: odds_score.points(odds_score.solo_skill(w, ref, h)[0]) for h in w.odds}
    assert skill["sharp"] > 5.0
    assert abs(skill["lookup"]) < 0.3
    assert skill["gates"] < -10.0
    assert odds_score.solo_skill(w, ref, "sharp")[1] == 10 * N_PAIRS


def test_a_model_the_head_has_no_row_for_is_scored_at_the_reference():
    w = world({"partial": only_strong, "full": sharp(0.05)}, epochs=range(6))
    ref = ref_of(w)
    su = odds_score.skill_units(w, ref, "partial")
    assert su.n_pairs == 6 * N_PAIRS
    # Zero on the pairs it does not cover, positive on the ones it does.
    for e in w.pairs:
        covered = np.array([m == "m/strong" for m in w.pairs[e].models])
        o = odds_score.head_odds_on(w, ref, "partial", e)
        assert np.array_equal(o[~covered], ref.of(e)[~covered].astype(np.int64))
    assert 0 < su.total < odds_score.skill_units(w, ref, "full").total


def test_a_head_is_scored_only_on_pairs_from_its_own_run():
    """Pairs from before a head served are published history it may have been
    fitted to: they are never read for it."""
    late = world({"old": sharp(0.05), "new": (sharp(0.05, seed=5), range(5, 8))}, epochs=range(8))
    ref = ref_of(late)
    assert odds_score.scored_epochs(late, "new") == [5, 6, 7]
    assert odds_score.skill_units(late, ref, "new").n_pairs == 3 * N_PAIRS
    assert odds_score.brier_units(late, ref, "new", "old").n_pairs == 3 * N_PAIRS
    # A miss breaks the run, as for net value: only the epochs since count.
    gap = world({"old": sharp(0.05), "new": (sharp(0.05), [0, 1, 2, 4, 5])}, epochs=range(6))
    assert odds_score.scored_epochs(gap, "new") == [4, 5]
    # The first-commit epoch bounds it too.
    assert odds_score.scored_epochs(late, "old", since_epoch=6) == [6, 7]


def test_loss_units_are_brier_points():
    assert odds_score.BRIER_UNITS_PER_POINT == 1_000_000
    p, y = np.array([ODDS_SCALE, 0, 5000], dtype=np.int16), np.array([0, 0, ODDS_SCALE], dtype=np.int32)
    assert odds_score.loss(p, y).tolist() == [ODDS_SCALE ** 2, 0, 5000 ** 2]
    assert odds_score.points(ODDS_SCALE ** 2) == 100.0


# --- the seat on odds ---------------------------------------------------------------


def step(w, state, heads, epoch=None, params=ODDS, **kw):
    epoch = max(w.slices) if epoch is None else epoch
    cands = {h: cand(h, seniority=i) for i, h in enumerate(heads)}
    return seat_step(state, epoch, f"e{epoch:08d}", w, cands, params=params, **kw)


def why(d, h):
    return [t for x, t in d.detail if x == h]


def test_an_empty_seat_goes_to_the_best_skill_not_to_the_most_accurate_router():
    w = world({"sharp": sharp(0.05), "blurry": sharp(0.25, seed=3), "lookup": lookup,
               "gates": gates}, epochs=range(6))
    d = step(w, SeatState(), ["gates", "lookup", "blurry", "sharp"])
    assert d.event == "first" and d.champion == "sharp"
    assert any("skill of its odds" in t for t in why(d, "sharp"))
    assert any("do not beat the reference" in t for t in why(d, "gates"))


def test_an_empty_seat_needs_enough_pairs():
    w = world({"sharp": sharp(0.05)}, epochs=range(4))          # 1,200 pairs
    d = step(w, SeatState(), ["sharp"])
    assert d.event == "empty" and any("only 1200 pairs; 1500 needed" in t for t in why(d, "sharp"))


def test_better_odds_take_the_seat_and_equal_or_lookup_odds_do_not():
    heads = {"champ": sharp(0.25, seed=1), "better": sharp(0.05, seed=2),
             "same": sharp(0.25, seed=1), "lookup": lookup}
    w = world(heads, epochs=range(6))
    d = step(w, SeatState(champion="champ", champion_since=0), list(heads))
    assert d.event == "take" and d.champion == "better"
    lines = why(d, "better")
    assert any("Brier gain=+" in t for t in lines) and any("skill=+" in t for t in lines)
    assert any(t.startswith("net value vs") for t in lines)
    # An identical head gains exactly nothing, and a table of rates loses to a head that reads.
    assert any("Brier gain=+0.000" in t for t in why(d, "same"))
    d2 = step(w, SeatState(champion="better", champion_since=0), list(heads))
    assert d2.event == "hold" and d2.champion == "better"


def test_a_take_needs_the_minimum_shared_pairs():
    heads = {"champ": sharp(0.25, seed=1), "better": sharp(0.05, seed=2)}
    w = world(heads, epochs=range(4))
    d = step(w, SeatState(champion="champ", champion_since=0), list(heads))
    assert d.event == "hold" and any("only 1200 shared pairs; 1500 needed" in t
                                     for t in why(d, "better"))


def test_sharper_odds_that_route_worse_do_not_take_the_seat():
    """The net-value guard: the new champion may not be a worse router."""
    heads = {"champ": sharp(0.25, seed=1), "better": sharp(0.05, seed=2)}
    w = world(heads, epochs=range(6), routing={"better": constant(0)})   # always the weak model
    d = step(w, SeatState(champion="champ", champion_since=0), list(heads))
    assert d.event == "hold"
    assert any("routes worse than" in t for t in why(d, "better"))


def test_beating_a_bad_champion_is_not_enough_without_beating_the_reference():
    """Switch-like odds beat worse switch-like odds by a wide margin and still
    do not take the seat: the odds must beat a table of rates."""
    def worse_gates(pairs, difficulty, epoch):
        return np.array([0 if m == "m/strong" else ODDS_SCALE for m in pairs.models], dtype=np.int16)
    w = world({"champ": worse_gates, "gates": gates}, epochs=range(6))
    # The champion was seated under net value; holding on odds is tested below.
    d = step(w, SeatState(champion="champ", champion_since=0), ["champ", "gates"])
    assert any("its odds do not beat the reference" in t for t in why(d, "gates"))
    assert d.champion != "gates"


def test_a_champion_whose_odds_are_worse_than_the_reference_does_not_hold():
    w = world({"gates": gates, "sharp": sharp(0.05)}, epochs=range(6))
    d = step(w, SeatState(champion="gates", champion_since=0), ["gates", "sharp"])
    assert any("champion odds are worse than the reference" in t for t in why(d, "gates"))
    assert d.state.champion != "gates"
    # A table of rates is not worse than the reference by more than noise, but it has
    # no margin either: a head that reads the question takes the seat from it.
    w2 = world({"lookup": lookup, "sharp": sharp(0.05)}, epochs=range(6))
    d2 = step(w2, SeatState(champion="lookup", champion_since=0), ["lookup", "sharp"])
    assert d2.champion == "sharp"


def test_confirmation_on_odds_reads_only_pairs_after_the_first_pass():
    confirm = dataclasses.replace(ODDS, confirm_questions=1500, confirm_open_epochs=5)
    heads = {"champ": sharp(0.25, seed=1), "better": sharp(0.05, seed=2)}
    state = SeatState(champion="champ", champion_since=0)        # settled
    w = world(heads, epochs=range(6))
    d = step(w, state, list(heads), params=confirm)
    assert d.event == "hold" and d.state.pending == (Pending("better", 5, "champ"),)
    assert any("passes again on 1500 pairs" in t for t in why(d, "better"))
    events = []
    for e in range(6, 12):
        w = world(heads, epochs=range(e + 1))
        d = step(w, d.state, list(heads), params=confirm)
        events.append(d.event)
        if d.event == "take":
            break
    # 300 pairs an epoch: the fifth fresh epoch is the first with 1,500.
    assert events == ["hold"] * 4 + ["take"]
    assert any("vs fresh pairs," in t and "pairs=1500" in t for t in why(d, "better"))
    assert d.state.confirm_from == 10


def test_the_decision_on_odds_does_not_depend_on_candidate_order():
    heads = {"champ": sharp(0.25, seed=1), "a": sharp(0.05, seed=2), "b": sharp(0.08, seed=3)}
    w = world(heads, epochs=range(6))
    state = SeatState(champion="champ", champion_since=0)
    one = step(w, state, ["champ", "a", "b"])
    cands = {h: cand(h, seniority=i) for i, h in enumerate(["champ", "a", "b"])}
    two = seat_step(state, 5, "e00000005", w, dict(reversed(list(cands.items()))), params=ODDS)
    assert one.state == two.state and one.detail == two.detail


def test_with_no_odds_parameters_the_window_extras_change_nothing():
    """Shadow: pairs and odds ride in the window while the seat is still net value."""
    heads = {"champ": sharp(0.25, seed=1), "better": sharp(0.05, seed=2)}
    with_odds = world(heads, epochs=range(6))
    plain = Window(with_odds.size, with_odds.slices, with_odds.heads)
    state = SeatState(champion="champ", champion_since=0)
    assert step(with_odds, state, list(heads), params=PARAMS) == step(plain, state, list(heads),
                                                                        params=PARAMS)


def test_window_rejects_odds_that_do_not_fit():
    sl, diff = make_slice(0)
    pairs, d = epoch_pairs(0)
    answers = {"a": HeadEpoch(np.zeros(Q, dtype=np.int64), np.zeros(Q, bool))}
    w = Window.empty(4)
    with pytest.raises(ValueError, match="do not align"):
        w.add_epoch(sl, answers, pairs, {"a": np.zeros(3, dtype=np.int16)})
    with pytest.raises(ValueError, match="no answers this epoch"):
        w.add_epoch(sl, answers, pairs, {"ghost": np.zeros(N_PAIRS, dtype=np.int16)})
    with pytest.raises(ValueError, match="without the epoch's pairs"):
        w.add_epoch(sl, answers, None, {"a": np.zeros(N_PAIRS, dtype=np.int16)})
    # The window drops pairs and odds with the epochs it drops.
    w = world({"a": sharp(0.05)}, epochs=range(6), size=4)
    assert sorted(w.pairs) == sorted(w.odds["a"]) == [2, 3, 4, 5]
