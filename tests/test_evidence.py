"""Tests for artifact-keyed evidence accumulation and the headroom score."""
from __future__ import annotations

import dataclasses

from fugal_subnet.config import BURN_IN_QUESTIONS
from fugal_subnet.evidence import (
    Evidence,
    _wilson_lower_bound,
    accumulate_epoch,
    apply_miss,
    decay_factor,
)
from fugal_subnet.scoring import burn_in_factor, composite

HALF_LIFE = 200


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

from fugal_subnet.frontier import Frontier  # noqa: E402

# A fixed, fully-measured frontier for these tests: cost per question on the
# x-axis, accuracy on the y-axis. Origin, a cheap model, a dear one.
FRONTIER = Frontier(points=((0.0, 0.0), (0.001, 0.5), (0.01, 0.8)), least_trials=1e6)


def _priced(ev, cost_per_question):
    n = ev.n_priced if ev.n_priced > 0 else ev.n_total
    return dataclasses.replace(ev, cost_sum=cost_per_question * n, n_priced=n)


def test_composite_is_headroom_ramped_and_confidence_scaled():
    ev = _priced(_epoch(None, n_correct=800, n_total=1000), 0.001)
    expected = (ev.wilson_lcb - FRONTIER.accuracy_at(0.001)) * burn_in_factor(ev.n_total)
    assert abs(composite(ev, FRONTIER) - expected) < 1e-12


def test_composite_uses_wilson_lcb_not_raw_accuracy():
    ev = _priced(_epoch(None, n_correct=8, n_total=10), 0.001)
    assert ev.wilson_lcb < ev.accuracy
    raw = (ev.accuracy - FRONTIER.accuracy_at(0.001)) * burn_in_factor(ev.n_total)
    assert composite(ev, FRONTIER) < raw


def test_a_constant_policy_on_the_frontier_scores_zero():
    """The reason the reference is the whole curve and not one model on it."""
    for cost, acc in FRONTIER.points[1:]:
        ev = Evidence(weights_hash="h", n_correct=acc * 1e6, n_total=1e6,
                      cost_sum=cost * 1e6, n_priced=1e6, pool_size=1e12)
        assert composite(ev, FRONTIER) == 0.0
    # And a point on the segment between them (a random mixture) too.
    ev = Evidence(weights_hash="h", n_correct=0.65 * 1e6, n_total=1e6,
                  cost_sum=0.0055 * 1e6, n_priced=1e6, pool_size=1e12)
    assert composite(ev, FRONTIER) == 0.0


def test_cheaper_at_equal_quality_scores_higher():
    """The incentive the subnet exists to create: the frontier is lower at a
    lower price, so the same accuracy is more headroom there."""
    expensive = Evidence("h1", n_correct=0.8 * 1e6, n_total=1e6, cost_sum=0.01 * 1e6,
                         n_priced=1e6, pool_size=1e12)
    frugal = Evidence("h2", n_correct=0.8 * 1e6, n_total=1e6, cost_sum=0.001 * 1e6,
                      n_priced=1e6, pool_size=1e12)
    assert composite(frugal, FRONTIER) > composite(expensive, FRONTIER) >= 0.0


def test_nothing_to_route_toward_scores_zero():
    """An empty frontier means no model answers anything — a real state, not a
    division to paper over. (Floor: 0 * anything is 0; headroom over an empty
    hull is the raw accuracy, but confidence is 0 with no trials.)"""
    ev = _priced(_epoch(None, n_correct=8, n_total=10), 0.001)
    assert composite(ev, Frontier(points=())) == 0.0


def test_burn_in_makes_penalty_washing_cost_what_earning_cost():
    """Reset clears penalties as readily as credit; the ramp is what prices it.

    Without a ramp, a miner with a poisoned record flips one weight bit and is
    immediately back at full score. With it, recovering costs the same evidence
    the position originally took.
    """
    good = None
    for _ in range(40):
        good = _epoch(good, n_correct=90, n_total=100, weights_hash="v1",
                      cost=0.1, ref_cost=0.3)
    established = composite(good, FRONTIER)
    assert established > 0

    poisoned = good
    for _ in range(20):
        poisoned = _epoch(poisoned, n_correct=5, n_total=100, weights_hash="v1",
                          cost=0.1, ref_cost=0.3)

    washed = _epoch(poisoned, n_correct=90, n_total=100, weights_hash="v2",
                    cost=0.1, ref_cost=0.3)

    assert washed.n_total == 100.0                        # accumulator did reset
    assert composite(washed, FRONTIER) < established      # but the score did not
    assert burn_in_factor(washed.n_total) < 0.2

    # And the ramp completes only after real work.
    assert burn_in_factor(BURN_IN_QUESTIONS) == 1.0


def test_wilson_lower_bound_edge_cases():
    assert _wilson_lower_bound(0.0, 0.0, 0.95) == 0.0
    assert _wilson_lower_bound(1.0, 1.0, 0.95) > 0.0
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) > 0.45
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) < 0.5
