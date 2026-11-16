"""Cost-aware exploration (protocol v3).

Exploration sends a question to a menu entry nobody chose. Drawn uniformly over
a 53-entry menu with budgets up to 128k tokens, a miner pays as much for one
Fable 5.2 @max call (~$5 at 128k) as for hundreds of cheap ones. So entries are
drawn with a probability that falls with their expected cost (never below a
floor of a uniform share), each pooled pair carries its inverse-probability
weight, and the odds score weights pairs by it: the seat test then estimates
what uniform sampling would have, without paying for it.
"""
import hashlib
import random
from fractions import Fraction

import numpy as np
import pytest

from fugal_subnet import explore_weights as xw

MENU = {  # entry: expected $ per call
    "cheap/a@low": Fraction(1, 10_000), "cheap/b@low": Fraction(3, 10_000),
    "mid/c@high": Fraction(4, 1000), "mid/d@high": Fraction(9, 1000),
    "dear/e@max": Fraction(1, 5), "dear/f@max": Fraction(1, 2),
}


def test_shares_fall_with_cost_and_never_below_the_floor():
    s = xw.shares(MENU, exponent_halves=2, floor=Fraction(1, 4))
    total = sum(s.values())
    m = len(MENU)
    assert s == xw.shares(dict(reversed(list(MENU.items()))), exponent_halves=2, floor=Fraction(1, 4))
    assert s["cheap/a@low"] > s["mid/c@high"] > s["dear/f@max"]
    for v in s.values():
        assert Fraction(v, total) >= Fraction(1, 4) / m - Fraction(1, 10**8)
    # Exponent 0 is uniform.
    u = xw.shares(MENU, exponent_halves=0, floor=Fraction(1, 4))
    assert len(set(u.values())) == 1


def test_weights_undo_the_sampling_and_are_bounded_by_the_floor():
    s = xw.shares(MENU, exponent_halves=2, floor=Fraction(1, 4))
    w = xw.weights(s)
    total, m = sum(s.values()), len(s)
    for e in s:
        # w = 1 / (M * pi), in WEIGHT_SCALE units.
        assert abs(w[e] - Fraction(total, m * s[e]) * xw.WEIGHT_SCALE) <= 1
        assert w[e] <= 4 * xw.WEIGHT_SCALE + 1
    # Expected weight under the sampling distribution is one.
    assert abs(sum(Fraction(s[e], total) * w[e] for e in s) - xw.WEIGHT_SCALE) <= m


def test_the_assignment_draws_each_entry_at_its_share():
    s = xw.shares(MENU, exponent_halves=2, floor=Fraction(1, 4))
    total = sum(s.values())
    counts = dict.fromkeys(s, 0)
    chosen = [{"question_id": f"q{i:03d}"} for i in range(32)]
    n = 0
    for k in range(3000):
        nonce = hashlib.sha256(f"n{k}".encode()).digest()
        a = xw.assign_by_share(nonce, chosen, s)
        assert a == xw.assign_by_share(nonce, list(reversed(chosen)), s)   # order-free
        for m in a.values():
            counts[m] += 1
            n += 1
    for e in s:
        p = s[e] / total
        assert abs(counts[e] / n - p) < 4 * (p * (1 - p) / n) ** 0.5 + 1e-3, (e, counts[e] / n, p)


def test_weighted_brier_estimates_the_uniform_sampling_score():
    """The quantity the seat compares is a mean per-pair Brier gain. Drawn
    uniformly, every entry counts equally. Drawn by cost with the weights, the
    mean matches that; without them it does not (cheap entries dominate)."""
    rng = random.Random(7)
    gain = {e: g for e, g in zip(MENU, (2.0, 1.5, 0.0, -1.0, -4.0, -6.0))}   # per-entry mean gain
    target = sum(gain.values()) / len(gain)                                 # uniform-sampling mean
    s = xw.shares(MENU, exponent_halves=2, floor=Fraction(1, 4))
    w = xw.weights(s)
    chosen = [{"question_id": f"q{i:03d}"} for i in range(32)]
    weighted, raw = [], []
    for k in range(400):
        a = xw.assign_by_share(hashlib.sha256(f"e{k}".encode()).digest(), chosen, s)
        us = np.array([gain[m] + rng.gauss(0, 1) for m in a.values()])
        ws = np.array([w[m] for m in a.values()], dtype=float) / xw.WEIGHT_SCALE
        weighted.append(float((us * ws).mean()))
        raw.append(float(us.mean()))
    assert abs(np.mean(weighted) - target) < 0.15, (np.mean(weighted), target)
    assert abs(np.mean(raw) - target) > 0.5                 # unweighted is biased toward cheap


def test_the_exponent_is_the_most_uniform_one_under_the_spend_cap():
    n = 32
    spend = {h: xw.expected_spend(xw.shares(MENU, exponent_halves=h, floor=Fraction(1, 4)), MENU, n)
             for h in range(5)}
    assert spend[0] > spend[2] > spend[4]                   # steeper is cheaper
    cap = (spend[1] + spend[2]) / 2
    h = xw.choose_exponent(MENU, n_calls=n, cap_usd=cap, floor=Fraction(1, 4), max_halves=4)
    assert h == 2 and spend[h] <= cap
    # Uniform when the cap allows it; the steepest when nothing fits.
    assert xw.choose_exponent(MENU, n_calls=n, cap_usd=spend[0], floor=Fraction(1, 4), max_halves=4) == 0
    assert xw.choose_exponent(MENU, n_calls=n, cap_usd=Fraction(0), floor=Fraction(1, 4), max_halves=4) == 4


def test_the_v3_menu_costs_come_from_the_pinned_tables():
    from fugal_subnet import routing_protocol
    costs = xw.entry_costs(routing_protocol.models_for_protocol(3))
    assert set(costs) == set(routing_protocol.models_for_protocol(3))
    assert costs["anthropic/claude-fable-5.2@max"] > costs["openai/gpt-6-luna@low"] > 0
    with pytest.raises(KeyError):
        xw.entry_costs(["not/a-menu-entry"])


# --- the weights travel with the pairs and weigh the odds score -------------------

def _answers():
    from fugal_subnet.explore_pool import ExploreAnswer
    return [
        ExploreAnswer("q1", "cheap/a@low", "ck1", True, False, True, 10, 20, latency_ms=900),
        ExploreAnswer("q1", "cheap/a@low", "ck2", True, False, True, 10, 22, latency_ms=1100),
        ExploreAnswer("q2", "dear/f@max", "ck1", False, False, True, 12, 4000, latency_ms=60_000),
    ]


def test_pooled_pairs_carry_their_weight_and_latency_into_the_reveal():
    from fugal_subnet import explore_pool
    w = {"cheap/a@low": 400_000, "dear/f@max": 4_000_000}
    pooled = explore_pool.pool_pairs(_answers(), nonce=b"n", coldkey_cap=36, weights=w)
    by = {p.model: p for p in pooled}
    assert by["cheap/a@low"].weight == 400_000 and by["dear/f@max"].weight == 4_000_000
    assert by["cheap/a@low"].latency_ms == 1000 and by["dear/f@max"].latency_ms == 60_000
    rows = explore_pool.pairs_for_reveal(pooled, {"q1": "mmlu", "q2": "aime"})
    assert [(r["weight"], r["latency_ms"]) for r in rows] == [(400_000, 1000), (4_000_000, 60_000)]
    # No weights given: every pair weighs one.
    plain = explore_pool.pool_pairs(_answers(), nonce=b"n", coldkey_cap=36)
    assert {p.weight for p in plain} == {xw.WEIGHT_SCALE}


def _window_with(weights):
    from fugal_subnet.champion import EpochPairs, EpochSlice, HeadEpoch, Window
    n = 4
    pairs = EpochPairs(
        question_ids=tuple(f"q{i}" for i in range(n)), models=("a", "a", "b", "b"),
        benchmarks=("mmlu",) * n, categories=("knowledge",) * n,
        y=np.array([10_000, 0, 10_000, 0], dtype=np.int32),
        prompt_tokens=np.zeros(n, dtype=np.int32), completion_tokens=np.zeros(n, dtype=np.int32),
        weight=None if weights is None else np.array(weights, dtype=np.int64))
    sl = EpochSlice(epoch=1, question_ids=("s0",), categories=("knowledge",), hard=(False,),
                    grades=np.ones((1, 2)), spends=np.ones((1, 2)))
    he = HeadEpoch(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=bool))
    # "sharp" is right on model a's pairs, R-like (0.5) on b's; "flat" says 0.5 everywhere.
    odds = {"sharp": np.array([10_000, 0, 5_000, 5_000], dtype=np.int16),
            "flat": np.array([5_000] * 4, dtype=np.int16)}
    return Window.empty(4).add_epoch(sl, {"sharp": he, "flat": he}, pairs, odds)


def test_the_odds_score_weighs_each_pair_by_its_weight():
    from fugal_subnet import odds_score
    plain = _window_with(None)
    ref = odds_score.reference(plain.pairs, 20)
    base = odds_score.brier_units(plain, ref, "sharp", "flat").total
    assert base == 2 * 5_000 ** 2                       # two pairs of model a, each 0.25 better
    heavy_a = _window_with([3 * xw.WEIGHT_SCALE, 3 * xw.WEIGHT_SCALE, xw.WEIGHT_SCALE, xw.WEIGHT_SCALE])
    assert odds_score.brier_units(heavy_a, ref, "sharp", "flat").total == 3 * base
    assert odds_score.skill_units(heavy_a, ref, "sharp").n_pairs == 4      # counts stay counts


def test_weights_survive_the_window_file_and_the_reveal(tmp_path):
    from fugal_subnet import explore_pool
    from fugal_subnet.champion import HeadEpoch
    from fugal_subnet.v2_epoch import load_window, save_window
    from fugal_subnet.v2_rebuild import pairs_from_reveal

    w = _window_with([2 * xw.WEIGHT_SCALE, xw.WEIGHT_SCALE, 3 * xw.WEIGHT_SCALE, xw.WEIGHT_SCALE])
    save_window(str(tmp_path / "w.npz"), w)
    back = load_window(str(tmp_path / "w.npz"), 4)
    assert back.pairs[1].weight.tolist() == w.pairs[1].weight.tolist()

    pooled = explore_pool.pool_pairs(_answers(), nonce=b"n", coldkey_cap=36,
                                     weights={"cheap/a@low": 400_000, "dear/f@max": 4_000_000})
    reveal = {"epoch_id": "e1", "exploration_pool": explore_pool.pairs_for_reveal(
        pooled, {"q1": "mmlu", "q2": "aime"}), "head_odds": {}}
    empty = HeadEpoch(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool))
    pairs, _ = pairs_from_reveal(reveal, {"h": empty})
    assert pairs.weight.tolist() == [400_000, 4_000_000]
    # A reveal from before weights existed reads as weight one.
    for r in reveal["exploration_pool"]:
        del r["weight"]
    pairs, _ = pairs_from_reveal(reveal, {})
    assert pairs.weight is None or set(pairs.weight.tolist()) == {xw.WEIGHT_SCALE}
