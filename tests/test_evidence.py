"""Tests for artifact-keyed evidence accumulation and the scoring formula."""
from __future__ import annotations

import dataclasses
import math

from fugal_subnet.config import BURN_IN_QUESTIONS, SCORE_QUALITY_EXPONENT
from fugal_subnet.evidence import (
    Evidence,
    _wilson_lower_bound,
    accumulate_epoch,
    apply_miss,
    decay_factor,
)
from fugal_subnet.scoring import burn_in_factor, composite, quality_term

HALF_LIFE = 200
ACC_BEST = 0.8


def _epoch(ev, *, n_correct=8, n_total=10, cost=0.5, ref_cost=0.3,
           weights_hash="abc123", pool_size=0.0):
    return accumulate_epoch(
        ev, weights_hash=weights_hash,
        n_correct=n_correct, n_total=n_total,
        cost=cost, ref_cost=ref_cost,
        half_life=HALF_LIFE, pool_size=pool_size,
    )


def test_fresh_accumulator():
    ev = _epoch(None)
    assert ev.weights_hash == "abc123"
    assert ev.n_correct == 8.0
    assert ev.n_total == 10.0
    assert ev.epochs_accumulated == 1
    assert ev.epochs_missed == 0
    assert ev.accuracy == 0.8
    assert 0.0 < ev.wilson_lcb < ev.accuracy


def test_same_hash_accumulates():
    ev1 = _epoch(None)
    ev2 = _epoch(ev1)
    assert ev2.epochs_accumulated == 2
    assert ev2.n_total > ev1.n_total
    assert ev2.wilson_lcb > ev1.wilson_lcb


def test_different_hash_resets():
    ev1 = _epoch(None, n_correct=10, n_total=10)
    ev_reset = _epoch(ev1, weights_hash="new_hash", n_correct=3, n_total=10)
    assert ev_reset.weights_hash == "new_hash"
    assert ev_reset.epochs_accumulated == 1
    assert ev_reset.n_correct == 3.0
    assert ev_reset.n_total == 10.0


def test_miss_drops_accuracy():
    ev = _epoch(None, n_correct=8, n_total=10)
    acc_before = ev.accuracy
    ev_missed = apply_miss(ev, n_expected=10, half_life=HALF_LIFE)
    assert ev_missed.accuracy < acc_before
    assert ev_missed.epochs_missed == 1
    assert ev_missed.n_total > ev.n_total


def test_miss_leaves_thrift_alone():
    """Skipping an epoch says nothing about how cheaply a miner routes."""
    ev = _epoch(None, cost=0.5, ref_cost=0.3)
    thrift_before = ev.thrift
    ev_missed = apply_miss(ev, n_expected=10, half_life=HALF_LIFE)
    assert abs(ev_missed.thrift - thrift_before) < 1e-12


def test_ewma_decay_geometric():
    alpha = decay_factor(HALF_LIFE)
    ev = _epoch(None, n_correct=10, n_total=10)
    for _ in range(9):
        ev = _epoch(ev, n_correct=10, n_total=10)

    expected_n = sum(alpha ** i * 10 for i in range(10))
    assert abs(ev.n_total - expected_n) < 0.01


def test_wilson_lcb_converges():
    ev1 = _epoch(None, n_correct=8, n_total=8)
    ev50 = ev1
    for _ in range(49):
        ev50 = _epoch(ev50, n_correct=8, n_total=8)
    assert ev50.wilson_lcb > ev1.wilson_lcb


def test_effective_n_is_capped_by_distinct_questions():
    """Wilson assumes independent trials; reused questions are not independent.

    At steady state n_total reaches ~86,550 over a ~21,000-question pool, which
    is ~4x reuse rather than 86,550 independent draws. Claiming the larger
    number overstates confidence.
    """
    pool = 1000.0
    ev = None
    for _ in range(300):
        ev = _epoch(ev, n_correct=80, n_total=100, pool_size=pool)
    assert ev.n_total > pool
    assert ev.effective_n == pool

    uncapped = dataclasses.replace(ev, pool_size=0.0)
    assert uncapped.effective_n > pool
    # Overstated confidence is a strictly tighter (higher) lower bound.
    assert uncapped.wilson_lcb > ev.wilson_lcb


def test_selective_publication_prevented():
    """A miner that only submits on good epochs should score lower
    than a consistent miner that never misses."""
    consistent = None
    selective = None
    for i in range(20):
        consistent = _epoch(consistent, n_correct=7, n_total=10,
                            weights_hash="consistent")
        if i % 2 == 0:
            selective = _epoch(selective, n_correct=10, n_total=10,
                               weights_hash="selective")
        else:
            if selective is not None:
                selective = apply_miss(selective, n_expected=10,
                                       half_life=HALF_LIFE)

    assert consistent is not None
    assert selective is not None
    assert consistent.wilson_lcb > selective.wilson_lcb


def test_serialization_roundtrip():
    ev = _epoch(None)
    ev = _epoch(ev)
    d = dataclasses.asdict(ev)
    restored = Evidence(**d)
    assert restored.weights_hash == ev.weights_hash
    assert abs(restored.n_correct - ev.n_correct) < 1e-10
    assert abs(restored.wilson_lcb - ev.wilson_lcb) < 1e-10
    assert restored.epochs_accumulated == ev.epochs_accumulated


# --- the scoring formula ---

def test_composite_is_the_weighted_geometric_mean():
    ev = _epoch(None, n_correct=800, n_total=1000, cost=0.5, ref_cost=0.3)
    w = SCORE_QUALITY_EXPONENT
    expected = (
        quality_term(ev.wilson_lcb, ACC_BEST) ** w
        * ev.thrift ** (1 - w)
    ) * burn_in_factor(ev.n_total)
    assert abs(composite(ev, ACC_BEST) - expected) < 1e-12


def test_composite_uses_wilson_lcb_not_raw_accuracy():
    ev = _epoch(None, n_correct=8, n_total=10)
    assert ev.wilson_lcb < ev.accuracy
    w = SCORE_QUALITY_EXPONENT
    raw = quality_term(ev.accuracy, ACC_BEST) ** w * ev.thrift ** (1 - w)
    assert composite(ev, ACC_BEST) < raw


def test_neither_axis_can_rescue_the_other():
    """The reason for a geometric mean rather than a weighted sum.

    Both degenerate strategies — perfect accuracy at any price, and near-zero
    cost at any accuracy — must score near zero. Under the old additive
    composite each collected its own term's weight regardless of the other.
    """
    ev = _epoch(None, n_correct=1000, n_total=1000, pool_size=1e9)
    accurate_but_ruinous = dataclasses.replace(ev, cost_sum=1e6, ref_cost_sum=0.3)
    cheap_but_wrong = dataclasses.replace(
        ev, n_correct=0.0, cost_sum=1e-9, ref_cost_sum=0.3,
    )

    assert composite(accurate_but_ruinous, ACC_BEST) < 0.05
    assert composite(cheap_but_wrong, ACC_BEST) == 0.0

    balanced = dataclasses.replace(ev, cost_sum=0.3, ref_cost_sum=0.3)
    assert composite(balanced, ACC_BEST) > composite(accurate_but_ruinous, ACC_BEST)
    assert composite(balanced, ACC_BEST) > composite(cheap_but_wrong, ACC_BEST)


def test_score_of_one_means_matched_the_reference_model():
    """The formula's headline claim, stated as a test."""
    ev = Evidence(
        weights_hash="h",
        n_correct=100_000.0, n_total=100_000.0,   # accuracy ~1.0, tight LCB
        cost_sum=1.0, ref_cost_sum=1.0,            # same cost as the reference
        pool_size=1e9,
    )
    # Matching the reference on both axes scores ~1.0.
    assert abs(composite(ev, acc_best=1.0) - 1.0) < 0.01
    # Beating it on cost scores above 1.0.
    cheaper = dataclasses.replace(ev, cost_sum=0.5)
    assert composite(cheaper, acc_best=1.0) > 1.0


def test_nothing_to_route_toward_scores_zero():
    """acc_best == 0 means no model answers anything — a real state, not a
    division to paper over."""
    ev = _epoch(None, n_correct=8, n_total=10)
    assert composite(ev, acc_best=0.0) == 0.0


def test_terms_are_capped():
    """A near-free model must not drive thrift to infinity.

    n_total is past the burn-in here on purpose, so the ramp is 1.0 and the cap
    is what is actually under test rather than the ramp masking it.
    """
    from fugal_subnet.config import (
        SCORE_QUALITY_CAP,
        SCORE_QUALITY_EXPONENT,
        SCORE_THRIFT_CAP,
    )

    ev = Evidence(
        weights_hash="h",
        n_correct=BURN_IN_QUESTIONS * 10.0, n_total=BURN_IN_QUESTIONS * 10.0,
        cost_sum=1e-12, ref_cost_sum=1e9,   # absurd thrift
        pool_size=1e12,
    )
    assert burn_in_factor(ev.n_total) == 1.0
    assert ev.thrift > SCORE_THRIFT_CAP

    w = SCORE_QUALITY_EXPONENT
    ceiling = SCORE_QUALITY_CAP ** w * SCORE_THRIFT_CAP ** (1 - w)
    assert composite(ev, acc_best=0.01) <= ceiling + 1e-9


def test_burn_in_makes_penalty_washing_cost_what_earning_cost():
    """Reset clears penalties as readily as credit; the ramp is what prices it.

    Without a ramp, a miner with a poisoned record flips one weight bit and is
    immediately back at full score. With it, recovering costs the same evidence
    the position originally took.
    """
    good = None
    for _ in range(40):
        good = _epoch(good, n_correct=90, n_total=100, weights_hash="v1",
                      cost=0.3, ref_cost=0.3)
    established = composite(good, ACC_BEST)

    poisoned = good
    for _ in range(20):
        poisoned = _epoch(poisoned, n_correct=5, n_total=100, weights_hash="v1",
                          cost=0.3, ref_cost=0.3)

    washed = _epoch(poisoned, n_correct=90, n_total=100, weights_hash="v2",
                    cost=0.3, ref_cost=0.3)

    assert washed.n_total == 100.0                      # accumulator did reset
    assert composite(washed, ACC_BEST) < established     # but the score did not
    assert burn_in_factor(washed.n_total) < 0.2

    # And the ramp completes only after real work.
    assert burn_in_factor(BURN_IN_QUESTIONS) == 1.0


def test_wilson_lower_bound_edge_cases():
    assert _wilson_lower_bound(0.0, 0.0, 0.95) == 0.0
    assert _wilson_lower_bound(1.0, 1.0, 0.95) > 0.0
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) > 0.45
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) < 0.5


def test_quality_exponent_satisfies_its_derivation():
    """w is derived from the product claim, not chosen — pin the derivation.

    "Match frontier quality at a fraction of the cost" makes quality a
    near-constraint: a router that gives up 40% of quality has not delivered
    the product however cheap it is, so it must not outscore simply matching
    the best model at the best model's price. That forces
    w > ln(6)/(ln(6)-ln(0.6)) = 0.778. An unweighted sqrt (w=0.5) fails it.
    """
    w_min = math.log(6) / (math.log(6) - math.log(0.6))
    assert SCORE_QUALITY_EXPONENT > w_min

    def score(quality, thrift):
        ev = Evidence(
            weights_hash="h",
            n_correct=quality * 1e6, n_total=1e6,
            cost_sum=1.0 / thrift, ref_cost_sum=1.0,
            pool_size=1e12,
        )
        return composite(ev, acc_best=1.0)

    matched_at_full_price = score(1.0, 1.0)
    lost_40pct_but_6x_cheaper = score(0.6, 6.0)
    the_product = score(1.0, 6.0)

    assert lost_40pct_but_6x_cheaper < matched_at_full_price
    assert the_product > matched_at_full_price
    # And 0.5 would have failed the same test.
    assert 0.6 ** 0.5 * 6 ** 0.5 > 1.0
