"""A head's odds are the same integer on every host (I1).

docs/INVARIANTS.md, "I1 — odds are computed exactly by the validator".
"""
import math
import random
from decimal import Decimal, localcontext

import numpy as np
import pytest

from fugal_subnet import head_odds
from fugal_subnet.head_odds import (
    HEAD_MAX_ABS_PARAM,
    ODDS_SCALE,
    check_embedding_bounds,
    check_head_bounds,
    logit_exact,
    odds,
    odds_of_logit,
    thresholds,
)


def test_the_threshold_table_is_pinned_and_does_not_depend_on_working_precision():
    table = thresholds()
    assert len(table) == ODDS_SCALE
    assert head_odds.thresholds_digest(table) == head_odds.THRESHOLDS_SHA256
    assert all(a < b for a, b in zip(table, table[1:]))
    for prec in (30, 100):
        with localcontext() as ctx:
            ctx.prec = prec
            other = tuple(float(((Decimal(k) + Decimal("0.5")) / ODDS_SCALE
                                 / (1 - (Decimal(k) + Decimal("0.5")) / ODDS_SCALE)).ln())
                          for k in range(0, ODDS_SCALE, 37))
        assert other == table[::37]


def test_a_different_table_is_refused(monkeypatch):
    monkeypatch.setattr(head_odds, "_THRESHOLDS", None)
    monkeypatch.setattr(head_odds, "THRESHOLDS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="differs from the pinned one"):
        thresholds()


def test_odds_are_the_logistic_rounded_to_the_nearest_ten_thousandth():
    assert odds_of_logit(0.0) == 5000
    assert odds_of_logit(-1e9) == 0 and odds_of_logit(1e9) == ODDS_SCALE
    assert odds_of_logit(float("-inf")) == 0 and odds_of_logit(float("inf")) == ODDS_SCALE
    rng = random.Random(4)
    for _ in range(2000):
        z = rng.uniform(-12, 12)
        p = 1 / (1 + math.exp(-z))
        # Within one unit of the libm value: equal except at a rounding boundary.
        assert abs(odds_of_logit(z) - round(p * ODDS_SCALE)) <= 1
    # Monotone, and symmetric about one half.
    zs = sorted(rng.uniform(-10, 10) for _ in range(500))
    ps = [odds_of_logit(z) for z in zs]
    assert ps == sorted(ps)
    with pytest.raises(ValueError):
        odds_of_logit(float("nan"))


def test_a_threshold_belongs_to_the_bin_above_it():
    table = thresholds()
    for k in (0, 1, 4999, 5000, 9999):
        assert odds_of_logit(table[k]) == k + 1
        assert odds_of_logit(math.nextafter(table[k], -math.inf)) == k


def test_the_logit_does_not_depend_on_the_order_of_the_terms():
    """The property a BLAS dot product does not have: permuting the terms, or
    summing them in blocks, cannot change the result."""
    rng = np.random.default_rng(11)
    w = (rng.standard_normal(1024) * 50).astype(np.float32)
    h = rng.standard_normal(1024).astype(np.float32)
    h /= np.linalg.norm(h)
    z = logit_exact(w, np.float32(0.25), h)
    for seed in range(20):
        perm = np.random.default_rng(seed).permutation(1024)
        assert logit_exact(w[perm], np.float32(0.25), h[perm]) == z
    # float32 products are exact in a double, so this is the true sum, rounded once.
    from fractions import Fraction
    exact = sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(w, h)),
                Fraction(float(np.float32(0.25))))
    assert z == float(exact)
    # A naive float32 dot product differs: that is why this module exists.
    assert float(np.dot(w, h) + np.float32(0.25)) != z


def test_cancelling_terms_do_not_lose_the_answer():
    w = np.array([1e6, 1.0, -1e6], dtype=np.float32)
    h = np.array([1.0, 1e-3, 1.0], dtype=np.float32)
    assert logit_exact(w, 0.0, h) == float(np.float32(1e-3))


def test_integer_and_float64_heads_are_scored_too():
    h = np.array([0.5, 0.25], dtype=np.float32)
    assert odds(np.array([2, -4], dtype=np.int64), 0, h) == 5000
    assert odds(np.array([2.0, -4.0], dtype=np.float64), 0.0, h) == 5000


def test_misaligned_inputs_are_refused():
    with pytest.raises(ValueError, match="do not align"):
        logit_exact(np.zeros(3), 0.0, np.zeros(4))


def test_head_odds_scores_each_pair_and_marks_models_the_head_lacks():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((2, 8)).astype(np.float32)
    b = np.array([0.1, -0.2], dtype=np.float32)
    H = rng.standard_normal((3, 8)).astype(np.float32)
    out = head_odds.head_odds(W, b, ["m/a", "m/b"], H, ["m/b", "m/zzz", "m/a"])
    assert out.dtype == np.int16
    assert out.tolist() == [odds(W[1], b[1], H[0]), -1, odds(W[0], b[0], H[2])]
    with pytest.raises(ValueError):
        head_odds.head_odds(W, b, ["m/a", "m/b"], H, ["m/a"])


def test_head_and_embedding_bounds():
    ok = np.zeros((2, 4), dtype=np.float32)
    check_head_bounds(ok, np.zeros(2))
    check_head_bounds(np.full((1, 4), HEAD_MAX_ABS_PARAM), np.zeros(1))
    for bad in (np.full((1, 4), HEAD_MAX_ABS_PARAM * 1.01), np.array([[np.inf, 0, 0, 0]]),
                np.array([[np.nan, 0, 0, 0]])):
        with pytest.raises(ValueError):
            check_head_bounds(bad, np.zeros(1))
    with pytest.raises(ValueError, match="b holds"):
        check_head_bounds(ok, np.array([0.0, -2e6]))
    check_embedding_bounds(np.full((3, 4), 0.5, dtype=np.float32))
    for bad in (np.full((1, 4), 1.5), np.array([[np.nan]])):
        with pytest.raises(ValueError):
            check_embedding_bounds(bad)
    # At the bounds every logit is finite: 1024 weights of 1e6 against unit entries.
    z = logit_exact(np.full(1024, HEAD_MAX_ABS_PARAM, dtype=np.float32), HEAD_MAX_ABS_PARAM,
                    np.ones(1024, dtype=np.float32))
    assert math.isfinite(z) and odds_of_logit(z) == ODDS_SCALE
