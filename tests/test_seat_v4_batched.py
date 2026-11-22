"""The seat's comparisons computed together (labels_v4.gain_strata_many,
seat_v4.decide's batches, seat_bound.bound_from_strata in integers) are
exactly what the per-pair code they replaced gave.

The reference is the per-pair path of the merged rules: `reference_decide`
is seat_v4.decide with `per_pair` (each comparison one labels_v4.gain_strata
call, which tests/test_labels_v4_vectorised.py holds to the per-label loops),
and `reference_bound` the Fraction bound. Compared: random strata; random
windows with every request shape (prefixes of one order as rule C asks,
one side fixed, scattered rows, repeats), heads across several row tiles,
successors' inherited rows, owners voting, labels no pair reads, empty
categories, values at the packed range's limit and past it, and any number
of threads; and every decision (state, evidence as published, order of its
keys, payable heads, rule C's tested lists) of the seat's synthetic world
through the scenarios of tests/test_seat_v4.py and a few hundred randomised
worlds (successors, ties in precedence, refused heads, missed epochs)."""
from __future__ import annotations

import dataclasses
import json
import random
import sys
from fractions import Fraction
from math import isqrt

import numpy as np
import pytest

from fugal_subnet import labels_v4, seat_bound, seat_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.labels_v4 import REFERENCE, Side
from fugal_subnet.seat_bound import Bound, Stratum
from tests.test_memo_v4_vectorised import _random_world

# --- the reference bound (seat_bound at d87deda) ---------------------------------------------


def reference_bound(by_category, *, margin_units, n_min, n0) -> Bound:
    a_alpha, ln2 = seat_bound.A_ALPHA, seat_bound.LN2_UP
    n = total = 0
    variance = Fraction(0)
    for _cat, s in by_category.items():
        cn, ct, cq = s.n, s.total, s.squares
        if cn < 0 or cq < 0 or cq * cn < ct * ct or (cn == 0 and (ct or cq)):
            raise ValueError("no set of items has these sums")
        if cn:
            variance += cq - Fraction(ct * ct, cn)
            n += cn
            total += ct
    if n == 0:
        return Bound(0, 0, variance, None, False)
    width = variance * Fraction(n + n0, n) * (a_alpha + ln2 * seat_bound.log_steps(n, n0))
    c = -(-width.numerator // width.denominator)
    r = isqrt(c)
    r = r if r * r >= c else r + 1
    lower = (total - r) // n
    d = total - margin_units * n
    return Bound(n, total, variance, lower, n >= n_min and d > 0 and d * d > width)


def test_the_integer_bound_is_the_fraction_bound():
    rng = random.Random(41)
    for _ in range(60_000):
        cats = {}
        for c in range(rng.randint(0, 6)):
            k = rng.choice([0, 1, 2, rng.randint(1, 4000)])
            mag = rng.choice([1, 10, 10**6, 10**9, 2**31, 10**19])
            xs = [rng.randint(-mag, mag) for _ in range(min(k, 40))]
            xs = (xs * (k // max(1, len(xs)) + 1))[:k]
            cats[f"c{c}"] = Stratum(len(xs), sum(xs), sum(x * x for x in xs))
        kw = dict(margin_units=rng.choice([-1_500_000, 0, 100_000, rng.randint(-10**8, 10**8)]),
                  n_min=rng.choice([0, 1, 1000, 1500]), n0=rng.choice([1, 7, 1000, 1500]))
        assert seat_bound.bound_from_strata(cats, **kw) == reference_bound(cats, **kw)


# --- gain_strata_many against gain_strata ------------------------------------------------------


def _requests(rng, sides, order):
    """Every shape the seat asks for: rule C's prefixes of one order (R
    first), one y against many x's, one x against scattered y's, repeats."""
    out = []
    for i, h in enumerate(order):
        out += [(sides[h], REFERENCE)] + [(sides[h], sides[k]) for k in order[:i]]
    bar = order[rng.integers(0, len(order))]
    out += [(sides[h], sides[bar]) for h in order]
    out += [(REFERENCE, sides[h]) for h in order]
    x = order[-1]
    scattered = [order[int(j)] for j in rng.permutation(len(order))]
    out += [(sides[x], sides[k]) for k in scattered] + out[:3]
    return out


def same_many(window, ref, requests, now, w=labels_v4.SEAT_WINDOW_EPOCHS, threads=1):
    try:
        want = [labels_v4.gain_strata(window, ref, x, y, now=now, window_epochs=w)
                for x, y in requests]
    except ValueError as e:
        with pytest.raises(ValueError, match=str(e).split(":")[0]):
            labels_v4.gain_strata_many(window, ref, requests, now=now, window_epochs=w,
                                       threads=threads)
        return None
    got = labels_v4.gain_strata_many(window, ref, requests, now=now, window_epochs=w,
                                     threads=threads)
    assert len(got) == len(want)
    for g, x in zip(got, want):
        assert g == dict(x)
        assert list(g) == sorted(g)
        assert all(type(v) is int for s in g.values() for v in (s.n, s.total, s.squares))
    return got


def test_random_windows_give_the_same_sums_for_every_request_shape():
    rng = np.random.default_rng(73)
    shown = {"empty": 0, "huge": 0, "several": 0}
    for k in range(300):
        window, _published, sides, now = _random_world(rng, k)
        ref = labels_v4.reference(window, now=now)
        w = labels_v4.SEAT_WINDOW_EPOCHS if k % 4 else int(rng.integers(1, 10))
        order = sorted(sides)
        got = same_many(window, ref, _requests(rng, sides, order), now, w, threads=1 + k % 4)
        if got:
            shown["empty"] += sum(not g for g in got)
            shown["several"] += sum(len(g) > 1 for g in got)
            shown["huge"] += sum(any(s.squares >= 2**63 for s in g.values()) for g in got)
    assert all(shown.values()), shown


def _big_world(rng, heads: int, epochs: int, *, labels=600, scale=None, weighted=True,
               successors=True, empty_category_epoch=True):
    """More heads than one row tile: heads with odds on a few epochs or all,
    successors (inherited rows), owners that vote, one epoch whose labels
    are all in one category (the others are empty there), and weights that
    put the packed value near its limit (`scale`)."""
    now = 200
    entries = [f"m/{i}" for i in range(6)]
    coldkeys = [f"ck{i}" for i in range(12)]
    pairs = {}
    for e in range(now - epochs + 1, now + 1):
        n = labels
        cats = (["only"] * n if empty_category_epoch and e == now - 1
                else [f"c{int(c)}" for c in rng.integers(0, 4, n)])
        y = rng.choice(np.array([0, 10_000, 5_000], np.int32), n, p=[0.45, 0.45, 0.1])
        weight = None
        if weighted:
            hi = scale or 3_000_000
            weight = rng.integers(max(1, hi // 3), hi + 1, n).astype(np.int64)
        base = EpochPairs(question_ids=tuple(f"q{e}_{i}" for i in range(n)),
                          models=tuple(entries[int(i)] for i in rng.integers(0, 6, n)),
                          benchmarks=tuple(f"b{c[-1]}" for c in cats), categories=tuple(cats),
                          y=y.astype(np.int32), prompt_tokens=np.zeros(n, np.int32),
                          completion_tokens=np.zeros(n, np.int32), weight=weight)
        voters = [tuple(sorted(set(rng.choice(coldkeys, int(rng.integers(1, 4))).tolist())))
                  for _ in range(n)]
        # missing: odds 0 (-1), an old allowance rate, or a provisional label
        pairs[e] = labels_v4.label_pairs(
            base, voters, rng.choice(np.array([-1, 5000, labels_v4.PROVISIONAL_LABEL], np.int16),
                                     n, p=[0.6, 0.2, 0.2]))
    odds, sides = {}, {}
    for j in range(heads):
        h = f"h{j:03d}"
        some = rng.random() < 0.4
        epochs_h = [e for e in pairs if not some or rng.random() < 0.3]
        odds[h] = {e: rng.choice(np.array([-1, 0, 1, 9_999, 10_000], np.int16), len(pairs[e].y))
                   if rng.random() < 0.2 else rng.integers(-1, 10_001, len(pairs[e].y)).astype(np.int16)
                   for e in epochs_h}
        since = int(rng.integers(now - epochs - 2, now + 1))
        rows = ({m: int(rng.integers(now - epochs - 5, since + 1)) for m in entries
                 if rng.random() < 0.5} if successors and rng.random() < 0.3 else {})
        chain = ()
        if successors and sides and rng.random() < 0.35:
            # a successor: earlier heads as its predecessors, each head of the
            # line until a later epoch (labels_v4.predecessor_odds reads their
            # stored odds where it has none of its own)
            preds = [str(p) for p in rng.choice(sorted(sides), min(len(sides), int(rng.integers(1, 4))),
                                                replace=False)]
            until = sorted((int(rng.integers(min(now - epochs, since), since + 1)) for _ in preds),
                           reverse=True)
            chain = tuple(zip(preds, until))
            for e in list(odds[h]):
                if rng.random() < 0.5:
                    del odds[h][e]            # read from a predecessor, if one has them
        added, line = frozenset(), None
        if successors and rng.random() < 0.4:
            # rows its line added (some inherited from an earlier adder) and the
            # line origin's commit, from which a missing row counts
            added = frozenset(m for m in entries if rng.random() < 0.3)
            line = int(rng.integers(now - epochs - 3, since + 1))
        sides[h] = Side(h, coldkeys[int(rng.integers(0, len(coldkeys)))], since, rows, chain,
                        added=added, line=line)
    return Window(size=48, slices={}, heads={}, pairs=pairs, odds=odds), sides, now


def test_added_rows_line_origins_and_provisional_labels_give_the_same_sums():
    """The merged label rules in the batched sums: a missing row and an added
    row's earlier labels from the line origin at odds 0, an added row from its
    own start, provisional labels read by nobody; two sides of one head that
    differ only in `added` or `line` are not confused."""
    rng = np.random.default_rng(29)
    shown = {"added": 0, "line": 0, "provisional": 0}
    for _k in range(60):
        window, sides, now = _big_world(rng, 12, 4, labels=240)
        ref = labels_v4.reference(window, now=now)
        order = list(sides)
        reqs = [(sides[h], sides[g]) for h in order for g in order if h != g]
        reqs += [(sides[h], REFERENCE) for h in order] + [(REFERENCE, sides[h]) for h in order]
        for h in order:
            s = sides[h]
            shown["added"] += bool(s.added)
            shown["line"] += s.line is not None
            if s.added or s.line is not None:
                bare = dataclasses.replace(s, added=frozenset(), line=None)
                reqs += [(bare, REFERENCE), (s, REFERENCE), (bare, s)]
        shown["provisional"] += sum(int((p.missing == labels_v4.PROVISIONAL_LABEL).any())
                                    for p in window.pairs.values())
        same_many(window, ref, reqs, now)
    assert all(v > 20 for v in shown.values()), shown


def test_successors_read_through_their_chain_give_the_same_sums():
    """A successor's epochs before its commit are read from its predecessors'
    stored odds (labels_v4.predecessor_odds) in both paths, and two sides of
    one head that differ only in their chain are not confused."""
    rng = np.random.default_rng(17)
    chained = 0
    for _k in range(40):
        window, sides, now = _big_world(rng, 14, 4, labels=200)
        ref = labels_v4.reference(window, now=now)
        order = list(sides)
        reqs = [(sides[h], sides[g]) for h in order for g in order if h != g]
        reqs += [(sides[h], REFERENCE) for h in order]
        for h in order:
            if sides[h].chain:
                chained += 1
                bare = dataclasses.replace(sides[h], chain=())
                reqs += [(bare, REFERENCE), (sides[h], REFERENCE), (REFERENCE, bare)]
        same_many(window, ref, reqs, now)
    assert chained > 20


def test_many_heads_across_row_tiles_give_the_same_sums():
    rng = np.random.default_rng(5)
    for k, (heads, epochs, scale) in enumerate(((70, 4, None), (45, 3, 20_000_000),
                                                 (40, 3, 25_000_000), (36, 2, None))):
        window, sides, now = _big_world(rng, heads, epochs, scale=scale, weighted=k != 3)
        ref = labels_v4.reference(window, now=now)
        order = list(sides)
        reqs = [(sides[h], REFERENCE) for h in order]
        for i, h in enumerate(order):
            reqs += [(sides[h], sides[g]) for g in order[:i]]
        reqs += [(sides[h], sides[order[0]]) for h in order[1:]]
        same_many(window, ref, reqs, now, threads=1 + k % 3)


def test_the_packed_range_holds_or_the_epoch_is_summed_the_old_way(monkeypatch):
    """A weight at which loss x weight passes 2^31 x WEIGHT_SCALE: that epoch
    goes through `_epoch_sums` (counted), the others are packed, and every
    sum is the same."""
    rng = np.random.default_rng(9)
    window, sides, now = _big_world(rng, 6, 3, weighted=True)
    e = sorted(window.pairs)[0]
    p = window.pairs[e]
    big = dataclasses.replace(p, weight=np.full(len(p.y), 30_000_000, np.int64))
    window = dataclasses.replace(window, pairs={**window.pairs, e: big})
    ref = labels_v4.reference(window, now=now)
    order = list(sides)
    reqs = [(sides[a], sides[b]) for a in order for b in order]
    want = [dict(labels_v4.gain_strata(window, ref, x, y, now=now)) for x, y in reqs]
    calls = []
    real = labels_v4._epoch_sums
    monkeypatch.setattr(labels_v4, "_epoch_sums", lambda *a: calls.append(a[4]) or real(*a))
    assert labels_v4.gain_strata_many(window, ref, reqs, now=now, threads=1) == want
    assert calls and set(calls) == {e}


def test_a_comparison_costs_a_fixed_number_of_python_lines_however_many_labels():
    """The operation count of gain_strata_many: per request, the lines run
    for an epoch of 2,000 labels are those for 500 (the labels are array
    work), and the requests share each head's per-epoch work."""
    from tests.test_v4_epoch_cost import count_lines

    def lines(n_labels, n_heads):
        rng = np.random.default_rng(n_labels)
        window, sides, now = _big_world(rng, n_heads, 3, labels=n_labels, successors=False,
                                        empty_category_epoch=False)
        ref = labels_v4.reference(window, now=now)
        order = list(sides)
        reqs = [(sides[h], sides[g]) for i, h in enumerate(order) for g in order[:i]]
        labels_v4.gain_strata_many(window, ref, reqs[:1], now=now, threads=1)   # the indexes
        _out, n = count_lines(lambda: labels_v4.gain_strata_many(window, ref, reqs, now=now,
                                                                 threads=1))
        return n

    small, large = lines(500, 12), lines(2000, 12)
    assert large <= small * 1.15 + 200, (small, large)


# --- decide against the previous decide -------------------------------------------------------


_REAL_DECIDE = seat_v4.decide


def reference_decide(state, **kw):
    """The per-pair path: seat_v4.decide with every rule A and C comparison
    its own labels_v4.gain_strata call (`per_pair`), on the merged rules
    (successors' predecessor chains, n0 = BOUND_N0, challenges, provisional
    labels, missing rows at odds 0 for the head's line)."""
    return _REAL_DECIDE(state, per_pair=True, **kw)


class Checked:
    """seat_v4.decide that also runs `reference_decide` and asserts every
    output is the same, the published evidence byte for byte."""

    def __init__(self, monkeypatch):
        self.calls = 0
        self.rule_c_rows = 0
        self.events = set()
        real = _REAL_DECIDE

        def decide(state, **kw):
            got = real(state, **kw)
            want = reference_decide(state, **kw)
            assert seat_v4.to_dict(got.state) == seat_v4.to_dict(want.state)
            assert (got.champion, got.event, got.bar, got.detail) == (want.champion, want.event,
                                                                      want.bar, want.detail)
            assert json.dumps(got.evidence) == json.dumps(want.evidence)
            assert seat_v4.digest(got.evidence) == seat_v4.digest(want.evidence)
            assert got.payable == want.payable
            assert list(got.tested.items()) == list(want.tested.items())
            self.calls += 1
            self.rule_c_rows += sum(len(d["earlier"]) for d in got.evidence["heads"].values())
            self.events.add(got.event)
            return got
        monkeypatch.setattr(seat_v4, "decide", decide)


def test_the_seat_scenarios_decide_as_before(monkeypatch):
    import inspect

    from tests import test_seat_v4 as scenarios

    checked = Checked(monkeypatch)
    for name, fn in sorted(vars(scenarios).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        if not params:
            fn()
        elif list(params) == ["missed"]:
            fn(1)
            fn(2)
    assert checked.calls > 300 and checked.rule_c_rows > 0
    assert {"fill", "take", "hold", "open"} <= checked.events


def _random_seat_world(rng: random.Random):
    """A World of 3 to 9 heads: skills spread or tied, shared owners and
    hotkeys (successors), equal blocks and blocks at the activation floor
    (precedence ties), missing rows."""
    from tests.seat_v4_world import ENTRIES, World

    floor = rng.choice([None, 6, 9])
    w = World(labels=rng.choice([160, 400, 900]), seed=rng.randrange(10**6), floor_block=floor,
              salt=f"s{rng.randrange(100)}")
    if rng.random() < 0.3:
        # the second entry joined provisional and became full at some epoch
        w.provisional_until[ENTRIES[1]] = rng.choice([2, 3])
    names = []
    for j in range(rng.randint(3, 9)):
        name = f"h{j}"
        kw = dict(skill=rng.choice([0.0, 0.2, 0.3, 0.3, 0.45, 0.6]),
                  block=rng.choice([3, 5, 5, 8, 11, 14]), noise=rng.choice([0, 0, j + 1]))
        if rng.random() < 0.25:
            kw["rows"] = (ENTRIES[0],)
        if rng.random() < 0.2:
            kw["owner"] = "ck-shared"
        w.add(name, **kw)
        names.append(name)
    return w, names


def test_hundreds_of_random_seat_worlds_decide_as_before(monkeypatch):
    from tests.seat_v4_world import ENTRIES, World  # noqa: F401  (loaded once)

    checked = Checked(monkeypatch)
    rng = random.Random(20261121)
    successions = 0
    for _k in range(200):
        w, names = _random_seat_world(rng)
        for e in range(1, rng.randint(6, 11)):
            if rng.random() < 0.15 and names:
                # a successor: the same hotkey and owner commit a head with an added row
                old = w.heads[rng.choice(names)]
                new = f"{old.name}s{e}"
                # a successor: the same rows, or the same rows and an added one
                rows = old.rows if rng.random() < 0.7 else ENTRIES
                w.add(new, owner=old.owner, hotkey=old.hotkey, block=e * 10 + 1, skill=old.skill,
                      rows=rows, digest_salt=old.digest_salt)
                names.append(new)
                successions += 1
            served = [n for n in names if rng.random() < 0.85]
            answered = [n for n in served if rng.random() < 0.8]
            value = {n: rng.choice([0.0, 0.0, 1.0, -2.0]) for n in answered}
            memo = [n for n in served if rng.random() < 0.05]
            w.epoch(e, served=served, answered=answered, value=value, memo=memo,
                    allow_take=rng.random() < 0.9)
    assert checked.calls >= 1000 and checked.rule_c_rows > 50 and successions > 10
    assert {"fill", "take", "hold", "empty"} <= checked.events


def test_the_timing_worlds_decide_as_before(monkeypatch):
    """The timing harness's world (scripts/experiments/time_v4_epoch.py) at
    small shapes: the adversarial ladder (rule C tests every earlier
    contender) and the honest field, both odds coverages, every head's
    precedence tied at the activation floor; the whole run_v2_epoch."""
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "experiments"))
    import time_v4_epoch as T

    from fugal_subnet.v2_epoch import run_v2_epoch

    checked = Checked(monkeypatch)
    for seed, ladder, coverage, uids in ((0, True, "realistic", 24), (1, True, "full", 16),
                                         (2, False, "realistic", 16), (3, True, "realistic", 40)):
        w = T.EpochWorld(T.Shape(uids=uids, epochs=6, dim=16, successors=2, returning=2,
                                 pool=6_000, reserve=900, seed=seed, ladder=ladder,
                                 coverage=coverage))
        run_v2_epoch(**w.run_kwargs())
    assert checked.calls == 4 and checked.rule_c_rows > 100
