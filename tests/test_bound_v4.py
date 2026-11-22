"""Protocol v4's one bound (docs/decisions/SEAT_MINIMAL_V4.md 3.1, 3.9, 10).

Every v4 seat comparison passes or fails on one function,
`champion.lower_bound`: a one-sided normal-mixture confidence sequence at
alpha = 1% on integer items, with the stratified variance estimated from the
items and the logarithm replaced by an integer upper bound. These tests hold
it to the design: the design's inequality exactly, exact integers and
Fractions only, the same answer in any order and in another process, never
tighter than the true mixture bound, the properties it claims, and the
simulated guarantees of 3.9 at a modest scale with a fixed seed.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import deque
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from fugal_subnet import champion, seat_bound
from fugal_subnet.seat_bound import A_ALPHA, LN2_UP, Bound, Stratum, bound_from_strata, lower_bound

ROOT = Path(__file__).resolve().parents[1]
UNIT = seat_bound.UNITS_PER_POINT


def _design_passes(items, margin_units, n_min, n0):
    """The inequality exactly as SEAT_MINIMAL_V4.md 3.1 writes it, computed
    here independently of the module."""
    xs = [x for v in items.values() for x in v]
    n = len(xs)
    if n < n_min:
        return False
    d = sum(xs) - margin_units * n
    if d <= 0:
        return False
    v = Fraction(0)
    for vals in items.values():
        if vals:
            v += sum(x * x for x in vals) - Fraction(sum(vals) ** 2, len(vals))
    rho = v * n0 / n
    k = -(-(n + n0) // n0)  # ceil(1 + n / n0)
    return d * d > (v + rho) * (Fraction(9211, 1000) + Fraction(6932, 10000) * k.bit_length())


def _random_items(rng: random.Random, n_cats=None, spread=8 * UNIT, mean=0, size=None):
    n_cats = n_cats or rng.randint(1, 7)
    out = {}
    for c in range(n_cats):
        m = size if size is not None else rng.randint(0, 60)
        out[f"cat{c}"] = [int(rng.gauss(mean, spread)) for _ in range(m)]
    return out


def _mixture_passes(items, margin_units, n_min, n0, prec=80):
    """The true one-sided normal-mixture boundary (Robbins; Howard et al.
    2021) at alpha = 1%, in high precision: D^2 > (V + rho)(2 ln(1/alpha) +
    ln((V + rho) / rho)), the logarithms not rounded."""
    xs = [x for v in items.values() for x in v]
    n = len(xs)
    if n < n_min or n == 0:
        return False
    d = sum(xs) - margin_units * n
    if d <= 0:
        return False
    v = Fraction(0)
    for vals in items.values():
        if vals:
            v += sum(x * x for x in vals) - Fraction(sum(vals) ** 2, len(vals))
    if v == 0:
        return True
    with localcontext() as ctx:
        ctx.prec = prec
        vrho = Decimal(v.numerator) / Decimal(v.denominator) * Decimal(n + n0) / Decimal(n)
        log_term = 2 * Decimal(100).ln() + (Decimal(n + n0) / Decimal(n0)).ln()
        return Decimal(d) * Decimal(d) > vrho * log_term


def _exact_lower(items, n0, prec=80) -> Decimal:
    """The real-valued lower bound of the mean, (S - sqrt(W)) / n, with the
    module's pinned rationals, in high precision."""
    xs = [x for v in items.values() for x in v]
    n = len(xs)
    v = Fraction(0)
    for vals in items.values():
        if vals:
            v += sum(x * x for x in vals) - Fraction(sum(vals) ** 2, len(vals))
    k = (-(-(n + n0) // n0)).bit_length()
    w = v * Fraction(n + n0, n) * (A_ALPHA + LN2_UP * k)
    with localcontext() as ctx:
        ctx.prec = prec
        return (Decimal(sum(xs)) - (Decimal(w.numerator) / Decimal(w.denominator)).sqrt()) / n


# --- the constants ---------------------------------------------------------


def test_the_constants_are_the_designs_rationals_and_never_below_the_logarithms():
    assert A_ALPHA == Fraction(9211, 1000)
    assert LN2_UP == Fraction(6932, 10000)
    assert UNIT == 1_000_000
    assert champion.lower_bound is lower_bound
    with localcontext() as ctx:
        ctx.prec = 60
        assert Decimal(9211) / 1000 >= 2 * Decimal(100).ln()
        assert Decimal(6932) / 10000 >= Decimal(2).ln()
        for n0 in (1, 7, 1000, 1500):
            for n in list(range(0, 200)) + [1499, 1500, 1501, 3000, 17_760, 96_000, 10**9, 10**15]:
                bound = LN2_UP * seat_bound.log_steps(n, n0)
                true = (Decimal(n + n0) / Decimal(n0)).ln()
                assert Decimal(bound.numerator) / Decimal(bound.denominator) >= true, (n, n0)


# --- exactness and determinism ---------------------------------------------


def test_the_result_holds_only_integers_fractions_and_a_bool():
    r = lower_bound({"a": [3, -1, 7], "b": [2, 2]}, margin_units=0, n_min=1, n0=2)
    assert isinstance(r, Bound)
    assert type(r.n) is int and type(r.total) is int and type(r.lower) is int
    assert isinstance(r.variance, Fraction) and type(r.passes) is bool
    assert (r.n, r.total) == (5, 13)


def test_the_variance_is_the_within_category_spread():
    r = lower_bound({"a": [3, -1, 7], "b": [2, 2]}, margin_units=0, n_min=1, n0=2)
    assert r.variance == (9 + 1 + 49 - Fraction(81, 3)) + (4 + 4 - Fraction(16, 2))
    # one category: the plain sum of squared deviations
    r = lower_bound({"a": [3, -1, 7, 2, 2]}, margin_units=0, n_min=1, n0=2)
    assert r.variance == 9 + 1 + 49 + 4 + 4 - Fraction(13 * 13, 5)


def test_passes_is_the_designs_inequality_exactly():
    rng = random.Random(11)
    seen = {True: 0, False: 0}
    for _ in range(600):
        items = _random_items(rng, mean=rng.choice([0, 2 * UNIT, 4 * UNIT]))
        margin = rng.choice([0, 100_000, -1_500_000, 1_000_000])
        n_min, n0 = rng.choice([(1, 1), (10, 10), (50, 30), (0, 1500)])
        got = lower_bound(items, margin_units=margin, n_min=n_min, n0=n0).passes
        assert got == _design_passes(items, margin, n_min, n0)
        seen[got] += 1
    assert seen[True] > 50 and seen[False] > 50


def test_the_same_items_give_the_same_result_in_any_key_and_item_order():
    rng = random.Random(5)
    for _ in range(80):
        items = _random_items(rng, mean=UNIT)
        base = lower_bound(items, margin_units=100_000, n_min=20, n0=40)
        keys = list(items)
        rng.shuffle(keys)
        shuffled = {}
        for k in keys:
            vals = list(items[k])
            rng.shuffle(vals)
            shuffled[k] = vals
        assert lower_bound(shuffled, margin_units=100_000, n_min=20, n0=40) == base
        # the items as any iterable, and their per-category sums
        gen = {k: iter(v) for k, v in reversed(list(shuffled.items()))}
        assert lower_bound(gen, margin_units=100_000, n_min=20, n0=40) == base
        assert bound_from_strata(seat_bound.strata(items), margin_units=100_000,
                                 n_min=20, n0=40) == base


def test_sums_kept_epoch_by_epoch_give_the_bound_of_the_items():
    rng = random.Random(9)
    epochs = [_random_items(rng, n_cats=3, mean=UNIT // 2) for _ in range(6)]
    pooled = {}
    window = {}
    for e in epochs:
        for k, v in e.items():
            pooled.setdefault(k, []).extend(v)
            window[k] = window.get(k, Stratum(0, 0, 0)) + seat_bound.strata({k: v})[k]
    for margin in (0, 100_000, UNIT // 2):
        assert (bound_from_strata(window, margin_units=margin, n_min=1, n0=50)
                == lower_bound(pooled, margin_units=margin, n_min=1, n0=50))
    # and an epoch leaves the window by subtraction
    first = seat_bound.strata(epochs[0])
    rest = {k: window[k] - first.get(k, Stratum(0, 0, 0)) for k in window}
    later = {}
    for e in epochs[1:]:
        for k, v in e.items():
            later.setdefault(k, []).extend(v)
    assert (bound_from_strata(rest, margin_units=0, n_min=1, n0=50)
            == lower_bound(later, margin_units=0, n_min=1, n0=50))


_CHILD = """
import json, random, sys
from fugal_subnet.seat_bound import lower_bound
rng = random.Random(3)
out = []
for _ in range(40):
    items = {f"c{c}": [int(rng.gauss(10**6, 8 * 10**6)) for _ in range(rng.randint(0, 50))]
             for c in range(rng.randint(1, 5))}
    r = lower_bound(items, margin_units=100_000, n_min=30, n0=50)
    out.append([r.n, r.total, str(r.variance), r.lower, r.passes])
print(json.dumps(out))
"""


def test_a_second_process_computes_the_same_result():
    env = dict(os.environ, PYTHONHASHSEED="4242")
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=120, check=True)
    rng = random.Random(3)
    here = []
    for _ in range(40):
        items = {f"c{c}": [int(rng.gauss(10**6, 8 * 10**6)) for _ in range(rng.randint(0, 50))]
                 for c in range(rng.randint(1, 5))}
        r = lower_bound(items, margin_units=100_000, n_min=30, n0=50)
        here.append([r.n, r.total, str(r.variance), r.lower, r.passes])
    assert json.loads(proc.stdout) == here


# --- the lower bound and the true mixture bound ----------------------------


def test_lower_is_the_exact_bound_rounded_down_and_the_float_formula_within_one_unit():
    rng = random.Random(21)
    for _ in range(300):
        items = _random_items(rng, mean=rng.choice([-UNIT, 0, UNIT]))
        n0 = rng.choice([1, 20, 1500])
        r = lower_bound(items, margin_units=0, n_min=0, n0=n0)
        if r.n == 0:
            assert r.lower is None
            continue
        exact = _exact_lower(items, n0)
        assert Decimal(r.lower) <= exact < Decimal(r.lower) + 1
        # the float formula a reader would write
        v = float(r.variance)
        k = (-(-(r.n + n0) // n0)).bit_length()
        lb = (r.total - ((v * (r.n + n0) / r.n) * (9.211 + 0.6932 * k)) ** 0.5) / r.n
        assert abs(r.lower - lb) <= 1 + 1e-9 * abs(lb)


def test_passes_exactly_when_the_exact_bound_is_above_the_margin():
    rng = random.Random(23)
    for _ in range(200):
        items = _random_items(rng, mean=UNIT, size=rng.randint(1, 40))
        r = lower_bound(items, margin_units=0, n_min=0, n0=10)
        exact = _exact_lower(items, 10)
        for margin in (r.lower - 1, r.lower, r.lower + 1):
            p = lower_bound(items, margin_units=margin, n_min=0, n0=10).passes
            assert p == (exact > margin)
        # the integer bound is never above a margin it fails
        assert lower_bound(items, margin_units=r.lower + 1, n_min=0, n0=10).passes is False


def test_it_is_never_tighter_than_the_true_mixture_bound():
    rng = random.Random(31)
    ours_only = wider = 0
    for _ in range(400):
        items = _random_items(rng, mean=rng.choice([0, UNIT, 3 * UNIT]),
                              size=rng.choice([None, 200]))
        n0 = rng.choice([1, 50, 1500])
        r = lower_bound(items, margin_units=0, n_min=0, n0=n0)
        if r.n == 0:
            continue
        # margins on and around the boundary, where a tighter bound would show
        for margin in (r.lower - 2, r.lower - 1, r.lower, r.lower + 1, 0, 100_000):
            ours = lower_bound(items, margin_units=margin, n_min=0, n0=n0).passes
            true = _mixture_passes(items, margin, 0, n0)
            if ours:
                assert true, (margin, n0)
            ours_only += ours
            wider += true and not ours
    assert ours_only > 100
    assert wider > 0  # the rounded logarithm does widen it somewhere


def test_from_the_minimum_evidence_it_widens_the_mixture_log_term_by_at_most_eight_percent():
    # Every rule's minimum is its n0, so a verdict is only ever read at
    # n >= n0, where the rounded logarithm adds at most about 7% to the log
    # term (3.5% to the width). Below it the term is at most 16% wider.
    with localcontext() as ctx:
        ctx.prec = 50
        for n0 in (1, 1000, 1500):
            for n in sorted({1, 2, n0 // 2, n0 - 1, n0, n0 + 1, 2 * n0 - 1, 2 * n0, 3 * n0,
                             10 * n0, 48 * 370, 10**9} - {0}):
                ours = A_ALPHA + LN2_UP * seat_bound.log_steps(n, n0)
                true = 2 * Decimal(100).ln() + (Decimal(n + n0) / Decimal(n0)).ln()
                ratio = Decimal(ours.numerator) / Decimal(ours.denominator) / true
                assert 1 <= ratio <= Decimal("1.08" if n >= n0 else "1.16"), (n, n0, ratio)


# --- what the design claims ------------------------------------------------


def test_a_mean_exactly_at_the_margin_never_passes():
    rng = random.Random(41)
    for _ in range(200):
        margin = rng.choice([0, 100_000, -1_500_000, 1_000_000])
        n = rng.randint(1, 3000)
        xs = [margin + int(rng.gauss(0, 8 * UNIT)) for _ in range(n - 1)]
        xs.append(margin * n - sum(xs))  # the mean is the margin exactly
        cats = {"a": xs[: n // 2], "b": xs[n // 2:]}
        assert lower_bound(cats, margin_units=margin, n_min=0, n0=1).passes is False
    # even with no variance at all
    assert lower_bound({"a": [100_000] * 5000}, margin_units=100_000, n_min=1,
                       n0=1500).passes is False


def test_fewer_than_the_minimum_never_passes():
    big = {"a": [50 * UNIT] * 1499}
    assert lower_bound(big, margin_units=100_000, n_min=1500, n0=1500).passes is False
    big["a"].append(50 * UNIT)
    assert lower_bound(big, margin_units=100_000, n_min=1500, n0=1500).passes is True
    rng = random.Random(43)
    for _ in range(50):
        items = _random_items(rng, mean=20 * UNIT, spread=UNIT)
        n = sum(len(v) for v in items.values())
        assert lower_bound(items, margin_units=0, n_min=n + 1, n0=10).passes is False


def test_adding_a_constant_shifts_the_bound_by_exactly_that_constant():
    rng = random.Random(47)
    for _ in range(100):
        items = _random_items(rng, mean=0)
        c = rng.randint(-5 * UNIT, 5 * UNIT)
        r = lower_bound(items, margin_units=0, n_min=0, n0=30)
        moved = {k: [x + c for x in v] for k, v in items.items()}
        s = lower_bound(moved, margin_units=0, n_min=0, n0=30)
        assert s.variance == r.variance
        if r.n:
            assert s.lower == r.lower + c
        for m in (-UNIT, 0, 100_000):
            assert (lower_bound(moved, margin_units=m + c, n_min=0, n0=30).passes
                    == lower_bound(items, margin_units=m, n_min=0, n0=30).passes)


def test_raising_a_category_a_lower_margin_or_a_lower_minimum_never_turns_a_pass_into_a_fail():
    rng = random.Random(53)
    checked = 0
    for _ in range(300):
        items = _random_items(rng, mean=3 * UNIT, size=rng.randint(5, 80))
        if not lower_bound(items, margin_units=100_000, n_min=10, n0=30).passes:
            continue
        checked += 1
        k = rng.choice(list(items))
        c = rng.randint(0, UNIT)
        up = dict(items, **{k: [x + c for x in items[k]]})
        assert lower_bound(up, margin_units=100_000, n_min=10, n0=30).passes
        assert lower_bound(items, margin_units=100_000 - rng.randint(1, UNIT),
                           n_min=10, n0=30).passes
        assert lower_bound(items, margin_units=100_000, n_min=rng.randint(0, 10),
                           n0=30).passes
    assert checked > 30


def test_scaling_items_and_margin_together_changes_no_verdict():
    rng = random.Random(59)
    for _ in range(150):
        items = _random_items(rng, mean=UNIT // 3, spread=UNIT)
        k = rng.randint(2, 1000)
        for m in (0, 100_000, -UNIT):
            scaled = {c: [k * x for x in v] for c, v in items.items()}
            assert (lower_bound(scaled, margin_units=k * m, n_min=5, n0=20).passes
                    == lower_bound(items, margin_units=m, n_min=5, n0=20).passes)


def test_more_of_the_same_evidence_never_lowers_the_bound():
    rng = random.Random(61)
    for _ in range(150):
        items = _random_items(rng, mean=rng.choice([0, UNIT]))
        n0 = rng.choice([1, 30, 1500])
        r = lower_bound(items, margin_units=0, n_min=0, n0=n0)
        if r.n == 0:
            continue
        doubled = {k: v + v for k, v in items.items()}
        assert lower_bound(doubled, margin_units=0, n_min=0, n0=n0).lower >= r.lower


# --- edge cases ------------------------------------------------------------


def test_zero_variance_gives_the_mean_itself():
    r = lower_bound({"a": [250_000] * 2000, "b": [-50_000] * 1000}, margin_units=100_000,
                    n_min=1500, n0=1500)
    assert r.variance == 0
    assert r.lower == (250_000 * 2000 - 50_000 * 1000) // 3000
    assert r.passes is True


def test_a_single_label():
    r = lower_bound({"a": [3 * UNIT]}, margin_units=100_000, n_min=1, n0=1500)
    assert (r.n, r.total, r.variance, r.lower, r.passes) == (1, 3 * UNIT, 0, 3 * UNIT, True)
    assert lower_bound({"a": [3 * UNIT]}, margin_units=100_000, n_min=2, n0=1500).passes is False
    # a stratum of one label carries no spread: the plug-in variance needs
    # strata that hold many labels, which the rules' minimums of 1,000 and
    # 1,500 on a handful of categories give
    many = {f"s{i}": [x] for i, x in enumerate([5 * UNIT, -UNIT, 2 * UNIT])}
    assert lower_bound(many, margin_units=0, n_min=1, n0=1).variance == 0


def test_no_items():
    r = lower_bound({}, margin_units=-UNIT, n_min=0, n0=1)
    assert (r.n, r.total, r.variance, r.lower, r.passes) == (0, 0, 0, None, False)
    r = lower_bound({"a": [], "b": []}, margin_units=-UNIT, n_min=0, n0=1)
    assert (r.n, r.lower, r.passes) == (0, None, False)


def test_huge_counts_and_values_stay_exact():
    # a trillion labels as sums, with values far past 64 bits
    n = 10**12
    s = Stratum(n, 3 * 10**5 * n, 64 * 10**12 * n + 9 * 10**10 * n)
    r = bound_from_strata({"a": s}, margin_units=100_000, n_min=1500, n0=1500)
    assert r.variance == 64 * 10**12 * n
    assert r.passes is True and 100_000 < r.lower < 300_000
    big = 10**30
    r = lower_bound({"a": [big, big + 2, big - 2]}, margin_units=big - 10, n_min=1, n0=1)
    assert r.total == 3 * big and r.variance == 8
    assert isinstance(r.lower, int)


def test_negative_gains():
    worse = {"a": [-2 * UNIT + (i % 7) * 1000 for i in range(3000)]}
    assert lower_bound(worse, margin_units=100_000, n_min=1500, n0=1500).passes is False
    r = lower_bound(worse, margin_units=-3 * UNIT, n_min=1500, n0=1500)
    assert r.passes is True and r.lower < -2 * UNIT + 4000
    # floor rounds toward minus infinity, never toward zero
    r = lower_bound({"a": [-7, -8]}, margin_units=-100, n_min=1, n0=10**9)
    assert r.lower <= _exact_lower({"a": [-7, -8]}, 10**9) < r.lower + 1


@pytest.mark.parametrize("bad", [
    {"a": [1.0]}, {"a": [True]}, {"a": [np.int64(3)]}, {"a": ["1"]}, {1: [1]},
])
def test_anything_but_integer_items_under_string_categories_is_refused(bad):
    with pytest.raises(TypeError):
        lower_bound(bad, margin_units=0, n_min=0, n0=1)


@pytest.mark.parametrize("kw", [
    {"margin_units": 0.5, "n_min": 0, "n0": 1}, {"margin_units": 0, "n_min": -1, "n0": 1},
    {"margin_units": 0, "n_min": 0, "n0": 0}, {"margin_units": True, "n_min": 0, "n0": 1},
    {"margin_units": 0, "n_min": 1.0, "n0": 1},
])
def test_bad_parameters_are_refused(kw):
    with pytest.raises((TypeError, ValueError)):
        lower_bound({"a": [1, 2]}, **kw)


@pytest.mark.parametrize("s", [Stratum(0, 1, 0), Stratum(-1, 0, 0), Stratum(2, 4, 7),
                               Stratum(1, 0, -1)])
def test_sums_no_set_of_items_could_have_are_refused(s):
    with pytest.raises(ValueError):
        bound_from_strata({"a": s}, margin_units=0, n_min=0, n0=1)


# --- the simulated guarantees of 3.9, at a modest scale ----------------------


def _seat_epochs(rng, d, sigma, per_epoch, epochs, window=48):
    """Rule A on its sliding 48-epoch window (margin 0.1, minimum 1,500,
    n0 = 1,500): the first epoch the bound passes, or None. Each epoch adds
    per_epoch normal labels of mean d and SD sigma (points), as integer
    units; the window is kept as per-epoch sums, exactly."""
    held: deque[Stratum] = deque()
    win = Stratum(0, 0, 0)
    for e in range(1, epochs + 1):
        x = np.rint(rng.normal(d, sigma, per_epoch) * UNIT).astype(np.int64)
        assert int(np.abs(x).max()) ** 2 * per_epoch < 2**63
        s = Stratum(per_epoch, int(x.sum()), int(np.dot(x, x)))
        held.append(s)
        win = win + s
        if len(held) > window:
            win = win - held.popleft()
        if bound_from_strata({"all": win}, margin_units=100_000, n_min=1500, n0=1500).passes:
            return e
    return None


@pytest.mark.parametrize("sigma,per_epoch,runs", [(4, 160, 60), (8, 370, 120), (12, 2000, 30)])
def test_heads_with_no_edge_never_take_over_300_epochs(sigma, per_epoch, runs):
    rng = np.random.default_rng(1000 + per_epoch)
    takes = [_seat_epochs(rng, 0.0, sigma, per_epoch, 300) for _ in range(runs)]
    assert takes.count(None) == runs


def test_thirty_heads_with_no_edge_over_300_epochs_none_seated():
    rng = np.random.default_rng(30)
    assert all(_seat_epochs(rng, 0.0, 8, 370, 300) is None for _ in range(30))


def test_a_head_at_the_margin_passes_at_most_one_and_a_half_percent_over_300_epochs():
    rng = np.random.default_rng(2026)
    runs = 600
    passed = sum(_seat_epochs(rng, 0.1, 8, 370, 300) is not None for _ in range(runs))
    # the design measured 1.0% at this setting (3.9); its test asks 1.5% at most
    assert passed / runs <= 0.015


# 3.9's speed table, the single bound: median epochs from the first label to
# a take, sigma 8. The design states no tolerance; these hold the median of
# a seeded sample to within max(1 epoch, 10%) of the design's figure.
@pytest.mark.parametrize("d,per_epoch,design", [
    (0.3, 370, 51), (0.5, 370, 15), (1.0, 370, 5),
    (0.3, 1000, 19), (0.5, 1000, 6), (1.0, 1000, 2),
])
def test_time_to_seat_matches_the_design(d, per_epoch, design):
    rng = np.random.default_rng(int(d * 10) * 10_000 + per_epoch)
    runs = 400
    times = sorted(_seat_epochs(rng, d, 8, per_epoch, 4 * design + 20) or 10**9
                   for _ in range(runs))
    median = times[runs // 2]
    assert abs(median - design) <= max(1, design // 10), (median, design)
