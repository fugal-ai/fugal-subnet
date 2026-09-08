"""The constant-policy frontier, and the properties the score must have.

The score is headroom above what any non-routing policy buys at the miner's
own price. Each test here is one property that, if it failed, would mean the
subnet pays for something other than routing:

  * every constant policy — one model, or a random mix of models — scores zero;
  * a router that beats the frontier at its price scores its headroom;
  * the frontier is a fact about the model pool, so no miner moves another's;
  * two validators build the same frontier from the same frame and slice;
  * a cold frame pays little, a measured one pays in full;
  * a miner that skips epochs does not thereby look cheaper.
"""
import dataclasses
import random

import pytest

from fugal_subnet.config import BURN_IN_QUESTIONS, FRONTIER_MIN_TRIALS, SCORE_QUALITY_FLOOR
from fugal_subnet.evidence import Evidence, accumulate_epoch, apply_miss
from fugal_subnet.frontier import Frontier, build_frontier, headroom
from fugal_subnet.reference_frame import ReferenceFrame, accumulate_exploration
from fugal_subnet.scoring import composite

# Five models: (rate_in, rate_out) per token and a true accuracy. Costs span
# ~50x, accuracies rise with price but with a dud in the middle (m/2 is dear
# for what it does) so the hull must skip a point.
PRICES = {
    "m/0": (1e-7, 2e-7), "m/1": (5e-7, 1e-6), "m/2": (2e-6, 4e-6),
    "m/3": (3e-6, 6e-6), "m/4": (5e-6, 1e-5),
}
TRUE_ACC = {"m/0": 0.30, "m/1": 0.55, "m/2": 0.50, "m/3": 0.75, "m/4": 0.85}
PROMPT_TOKENS = 100 * 300      # 300 questions of 100 tokens
N_Q = 300
COMPLETION = 300.0


def _warm_frame(trials_per_model=400, seed=0):
    rng = random.Random(seed)
    frame = ReferenceFrame()
    samples = [(m, rng.random() < TRUE_ACC[m], 100, int(COMPLETION))
               for m in PRICES for _ in range(trials_per_model)]
    return accumulate_exploration(frame, samples)


def _cost_per_q(model):
    p_in, p_out = PRICES[model]
    return (p_in * PROMPT_TOKENS + p_out * COMPLETION * N_Q) / N_Q


def _evidence(acc, cost_per_q, n=BURN_IN_QUESTIONS * 4, weights_hash="h"):
    return Evidence(weights_hash=weights_hash, n_correct=acc * n, n_total=float(n),
                    cost_sum=cost_per_q * n, ref_cost_sum=0.0, pool_size=1e9,
                    n_priced=float(n))


def _frontier(frame=None):
    return build_frontier(frame or _warm_frame(), PRICES, PROMPT_TOKENS, N_Q, COMPLETION)


def test_hull_is_concave_starts_at_origin_and_skips_dominated_models():
    f = _frontier()
    assert f.points[0] == (0.0, 0.0)
    costs = [c for c, _ in f.points]
    assert costs == sorted(costs)
    # Concave: slopes non-increasing.
    slopes = [(a1 - a0) / (c1 - c0) for (c0, a0), (c1, a1) in zip(f.points, f.points[1:])]
    assert all(s1 <= s0 + 1e-12 for s0, s1 in zip(slopes, slopes[1:]))
    # m/2 costs more than m/1 for less accuracy than the m/1→m/3 line: not on the hull.
    assert "m/2" not in f.models
    assert f.max_accuracy == pytest.approx(f.accuracy_at(1e9))


def test_every_constant_policy_scores_zero():
    """One model, always: the whole reason for this scoring."""
    frame = _warm_frame()
    f = _frontier(frame)
    for m in PRICES:
        ev = _evidence(frame.accuracy(m), _cost_per_q(m))
        # Its lower bound sits below the hull at its own cost — headroom <= 0.
        assert headroom(ev.wilson_lcb, ev.cost_per_question, f) <= 1e-9
        assert composite(ev, f) == 0.0


def test_every_random_mixture_of_two_models_scores_zero():
    """A coin flip between two models reads nothing either."""
    frame = _warm_frame()
    f = _frontier(frame)
    models = sorted(PRICES)
    for a in models:
        for b in models:
            for p in (0.25, 0.5, 0.75):
                acc = p * frame.accuracy(a) + (1 - p) * frame.accuracy(b)
                cost = p * _cost_per_q(a) + (1 - p) * _cost_per_q(b)
                ev = _evidence(acc, cost)
                assert composite(ev, f) == 0.0, (a, b, p)


def test_a_router_above_the_frontier_scores_its_headroom():
    frame = _warm_frame()
    f = _frontier(frame)
    cost = _cost_per_q("m/1")                     # pays what m/1 costs...
    ref = f.accuracy_at(cost)
    ev = _evidence(ref + 0.12, cost)              # ...answers 12 points better
    assert composite(ev, f) > 0
    assert composite(ev, f) == pytest.approx((ev.wilson_lcb - ref) * f.confidence, abs=1e-9)
    # More headroom, more score; same headroom at a lower price is not penalised.
    better = _evidence(ref + 0.20, cost)
    assert composite(better, f) > composite(ev, f)


def test_headroom_is_largest_where_the_frontier_is_low_and_steep():
    """Cheap-and-smart is the product. A router matching m/3's accuracy at
    m/0's price earns far more than one matching m/4's at m/4's price."""
    frame = _warm_frame()
    f = _frontier(frame)
    cheap_smart = _evidence(frame.accuracy("m/3"), _cost_per_q("m/0"))
    dear_matching = _evidence(frame.accuracy("m/4") + 0.02, _cost_per_q("m/4"))
    assert composite(cheap_smart, f) > 3 * composite(dear_matching, f)


def test_quality_floor_removes_routers_far_below_the_best_model():
    frame = _warm_frame()
    f = _frontier(frame)
    tiny_cost = _cost_per_q("m/0") * 0.5
    # Beats the (very low) frontier at a tiny price, but is far below the best model.
    low = _evidence(SCORE_QUALITY_FLOOR * f.max_accuracy - 0.05, tiny_cost)
    assert headroom(low.wilson_lcb, low.cost_per_question, f) > 0
    assert composite(low, f) == 0.0
    ok = _evidence(SCORE_QUALITY_FLOOR * f.max_accuracy + 0.05, tiny_cost)
    assert composite(ok, f) > 0


def test_no_miner_moves_another_miners_reference():
    """I4: the frontier is built from the frame and the slice, never from heads.
    Adding, removing or copying miners changes nothing about it."""
    frame = _warm_frame()
    f_alone = _frontier(frame)
    # "Other miners" can only influence the frame through exploration samples,
    # which are nonce-assigned; simulate a large field's honest samples and a
    # sybil's all-wrong samples on top of a well-measured frame.
    rng = random.Random(1)
    field = [(m, rng.random() < TRUE_ACC[m], 100, 300) for m in PRICES for _ in range(30)]
    sybil = [(m, False, 100, 300) for m in PRICES for _ in range(15)]
    f_field = _frontier(accumulate_exploration(frame, field))
    f_sybil = _frontier(accumulate_exploration(frame, sybil))
    probe = _cost_per_q("m/1")
    assert abs(f_field.accuracy_at(probe) - f_alone.accuracy_at(probe)) < 0.02
    assert abs(f_sybil.accuracy_at(probe) - f_alone.accuracy_at(probe)) < 0.02


def test_two_validators_build_identical_frontiers():
    """I1/I9: same frame, same slice, same prices → same bytes."""
    frame = _warm_frame(seed=3)
    a = build_frontier(frame, PRICES, PROMPT_TOKENS, N_Q, COMPLETION)
    b = build_frontier(ReferenceFrame.from_dict(frame.to_dict()), dict(reversed(list(PRICES.items()))),
                       PROMPT_TOKENS, N_Q, COMPLETION)
    assert a == b


def test_cold_frame_pays_little_and_a_measured_one_pays_in_full():
    cold = build_frontier(ReferenceFrame(), PRICES, PROMPT_TOKENS, N_Q, COMPLETION)
    assert cold.confidence == 0.0
    # With no evidence every model is the prior, so "headroom" over it is not
    # routing — and it is paid nothing.
    ev = _evidence(0.9, _cost_per_q("m/1"))
    assert headroom(ev.wilson_lcb, ev.cost_per_question, cold) > 0
    assert composite(ev, cold) == 0.0
    warm = _frontier(_warm_frame(trials_per_model=int(FRONTIER_MIN_TRIALS) + 10))
    assert warm.confidence == 1.0
    half = _frontier(_warm_frame(trials_per_model=int(FRONTIER_MIN_TRIALS // 2)))
    assert 0.3 < half.confidence < 0.7


def test_skipping_epochs_does_not_make_a_miner_look_cheaper():
    ev = accumulate_epoch(None, "h", n_correct=240, n_total=300, cost=0.30,
                          ref_cost=0.0, half_life=200, pool_size=1e9)
    before = ev.cost_per_question
    missed = apply_miss(ev, n_expected=300, half_life=200)
    assert missed.n_total > ev.n_total
    assert missed.cost_per_question == pytest.approx(before)


def test_old_state_without_n_priced_still_loads_and_prices():
    """State files written before n_priced existed must still load; the cost
    per question then falls back to n_total, which is the old behaviour."""
    ev = Evidence(weights_hash="h", n_correct=8.0, n_total=10.0, cost_sum=1.0)
    assert ev.n_priced == 0.0
    assert ev.cost_per_question == pytest.approx(0.1)
    d = dataclasses.asdict(ev)
    d.pop("n_priced")
    assert Evidence(**d).cost_per_question == pytest.approx(0.1)


def test_migrated_evidence_is_not_priced_at_a_history_of_cost_over_one_epoch():
    """Measured on testnet 552, e00022118: a record written before n_priced
    existed carried three epochs of decayed cost with n_priced == 0; the first
    new epoch decayed the zero, added 300, and divided the whole history by
    300 — 4.4x the miner's real cost per question, a third of its headroom.
    A pre-n_priced record priced every question it counted."""
    old = Evidence(weights_hash="h", n_correct=700.0, n_total=900.0, cost_sum=0.90,
                   ref_cost_sum=0.0, pool_size=1e9)          # 3 epochs at $0.001/q
    assert old.n_priced == 0.0 and old.cost_per_question == pytest.approx(0.001)
    new = accumulate_epoch(old, "h", n_correct=240, n_total=300, cost=0.30,
                           ref_cost=0.0, half_life=200, pool_size=1e9)
    assert new.cost_per_question == pytest.approx(0.001, rel=1e-6)
    missed = apply_miss(old, n_expected=300, half_life=200)
    assert missed.cost_per_question == pytest.approx(0.001, rel=1e-6)


def test_the_unconfident_share_of_emission_burns_after_normalisation():
    """Measured on testnet 552, e00022118: the frontier had confidence 0.08 and
    the one verified miner received 100% of the miner share, because a factor
    common to every score cancels when the scores are normalised. The burn is
    applied in compute_weights, after normalising."""
    from fugal_subnet.rewards import BURN_UID, compute_weights
    from fugal_subnet.scoring import MinerRecord
    recs = {
        5: MinerRecord(uid=5, hotkey="a", composite_score=0.0038),
        6: MinerRecord(uid=6, hotkey="b", composite_score=0.0010),
    }
    uids, w = compute_weights(recs, paid_fraction=0.08)
    got = dict(zip(uids, w))
    assert got[BURN_UID] == pytest.approx(0.92)
    assert got[5] == pytest.approx(0.08 * 0.0038 / 0.0048)
    assert got[6] == pytest.approx(0.08 * 0.0010 / 0.0048)
    assert sum(w) == pytest.approx(1.0)
    # Full confidence: nothing burns, ranking unchanged.
    uids, w = compute_weights(recs, paid_fraction=1.0)
    assert BURN_UID not in uids and dict(zip(uids, w))[5] == pytest.approx(0.0038 / 0.0048)


def test_frontier_is_flat_past_the_most_accurate_model():
    f = _frontier()
    top = max(_cost_per_q(m) for m in PRICES)
    assert f.accuracy_at(top * 10) == f.accuracy_at(top)


def test_empty_frontier_is_harmless():
    f = Frontier(points=())
    assert f.accuracy_at(1.0) == 0.0 and f.max_accuracy == 0.0
