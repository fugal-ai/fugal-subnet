"""Tests for artifact-keyed evidence accumulation."""
from __future__ import annotations

import dataclasses

from fugal_subnet.evidence import (
    Evidence,
    _wilson_lower_bound,
    accumulate_epoch,
    apply_miss,
    decay_factor,
)
from fugal_subnet.scoring import _composite_from_evidence, _normalize_kl

HALF_LIFE = 200


def _epoch(ev, *, n_correct=8, n_total=10, head_cost=0.5, oracle_cost=0.3,
           kl_total=0.2, n_kl=10, coverage=1.0, weights_hash="abc123"):
    return accumulate_epoch(
        ev, weights_hash=weights_hash,
        n_correct=n_correct, n_total=n_total,
        head_cost=head_cost, oracle_cost=oracle_cost,
        kl_total=kl_total, n_kl=n_kl,
        coverage=coverage, half_life=HALF_LIFE,
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


def test_composite_uses_wilson_lcb():
    ev = _epoch(None, n_correct=8, n_total=10, head_cost=0.5,
                oracle_cost=0.3, kl_total=0.2, n_kl=10, coverage=1.0)
    composite = _composite_from_evidence(ev)
    assert composite > 0
    raw_acc_composite = 0.55 * ev.accuracy + 0.35 * ev.cost_efficiency + 0.10 * _normalize_kl(ev.avg_kl)
    wilson_composite = 0.55 * ev.wilson_lcb + 0.35 * ev.cost_efficiency + 0.10 * _normalize_kl(ev.avg_kl)
    assert abs(composite - wilson_composite) < 1e-10
    assert composite < raw_acc_composite


def test_half_life_decay():
    alpha = decay_factor(HALF_LIFE)
    decayed = 100.0
    for _ in range(HALF_LIFE):
        decayed *= alpha
    assert abs(decayed - 50.0) < 0.1


def test_wilson_lower_bound_edge_cases():
    assert _wilson_lower_bound(0.0, 0.0, 0.95) == 0.0
    assert _wilson_lower_bound(1.0, 1.0, 0.95) > 0.0
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) > 0.45
    assert _wilson_lower_bound(0.5, 1000.0, 0.95) < 0.5
