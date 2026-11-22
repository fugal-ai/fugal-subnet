"""The memorisation split, computed with integer arrays (memo_v4._Split),
gives exactly what the per-label loop it replaced gave: the same parts, the
same integers and Fractions, the same verdicts and the same reveal bytes.

The loop is kept here as the reference (`reference_split_labels`,
`reference_evidence_from_parts`, `reference_verdicts`: the module's code at
923170e, verbatim but for names) and compared on the memo tests' windows,
the published 44-epoch reproduction, the seat-memo world, and a few hundred
randomised windows (successors with inherited rows, heads with no labels on
one side, benchmarks left out, owners voting, missing odds, unweighted and
weighted labels, skill sums too large for int64, empty reserves)."""
from __future__ import annotations

import json
import math
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

from fugal_subnet import labels_v4, memo_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.labels_v4 import Side
from fugal_subnet.memo_v4 import (
    MEMO_MARGIN_UNITS,
    MEMO_MIN_LABELS,
    MEMO_MIN_QUESTIONS,
    SEEN,
    UNSEEN,
    MemoEvidence,
    Part,
    Published,
)
from fugal_subnet.seat_bound import Stratum, bound_from_strata, int_sum, int_sum_squares

# --- the reference: memo_v4 at 923170e --------------------------------------------------------


def reference_split_labels(window, ref, side, published, *, now, window_epochs=None):
    from fugal_subnet.explore_weights import apply

    if side.head is None:
        raise ValueError("the memorisation check is of a head, not of R")
    w = labels_v4.SEAT_WINDOW_EPOCHS if window_epochs is None else int(window_epochs)
    published.require(int(now) - 1)
    acc: dict = {}
    for e in sorted(window.pairs):
        if not labels_v4.in_window(e, now, w):
            continue
        pairs = window.pairs[e]
        ox = labels_v4._scored(window, ref, side, e)
        if ox is None:
            continue
        keep = reference_eligible(pairs, e, side, labels_v4.REFERENCE, now=now, window_epochs=w)
        if not keep.any():
            continue
        oy = labels_v4._scored(window, ref, labels_v4.REFERENCE, e)
        t = pairs.y.astype(np.int64)
        units = (oy - t) ** 2 - (ox - t) ** 2
        if pairs.weight is not None:
            units = apply(units, pairs.weight)
        for i in np.flatnonzero(keep).tolist():
            q, m, b = pairs.question_ids[i], pairs.models[i], str(pairs.benchmarks[i])
            f = published.first.get(q)
            part = SEEN if f is not None and f < side.row_since(m) else UNSEEN
            cell = acc.setdefault(b, {SEEN: {}, UNSEEN: {}})[part].setdefault(q, [0, 0])
            cell[0] += 1
            cell[1] += int(units[i])
    return {b: {p: Part(sum(c[0] for c in parts[p].values()),
                        MappingProxyType({q: (c[0], c[1]) for q, c in sorted(parts[p].items())}))
                for p in (SEEN, UNSEEN)}
            for b, parts in sorted(acc.items())}


def reference_eligible(pairs, epoch, x, y, *, now, window_epochs=labels_v4.SEAT_WINDOW_EPOCHS):
    if not isinstance(pairs, labels_v4.LabelPairs):
        raise ValueError(f"epoch {epoch} is not a v4 epoch: its labels carry no voters")
    n = len(pairs.question_ids)
    if not labels_v4.in_window(epoch, now, window_epochs):
        return np.zeros(n, dtype=bool)
    owners = {s.owner for s in (x, y) if s.owner is not None}
    keep = np.ones(n, dtype=bool)
    for i, (entry, voters) in enumerate(zip(pairs.models, pairs.voters)):
        for side in (x, y):
            start = side.row_since(entry)
            if start is not None and epoch < start:
                keep[i] = False
        if owners.intersection(voters):
            keep[i] = False
    return keep


def reference_evidence_from_parts(parts) -> MemoEvidence:
    kept = {b: (p[SEEN], p[UNSEEN]) for b, p in sorted(parts.items())
            if len(p[SEEN].by_question) >= MEMO_MIN_QUESTIONS
            and len(p[UNSEEN].by_question) >= MEMO_MIN_QUESTIONS}
    left_out = tuple(sorted(set(parts) - set(kept)))
    n_s = sum(s.labels for s, _ in kept.values())
    n_u = sum(u.labels for _, u in kept.values())
    if not kept:
        return MemoEvidence(MappingProxyType({}), n_s, n_u, Fraction(0), None, None, False, left_out)

    def total(part):
        return sum(u for _, u in part.by_question.values())

    w = {b: Fraction(s.labels * u.labels, s.labels + u.labels) for b, (s, u) in kept.items()}
    big_w = sum(w.values())
    coef = {}
    for b, (s, u) in kept.items():
        coef[(b, SEEN)] = w[b] / (big_w * s.labels)
        coef[(b, UNSEEN)] = -w[b] / (big_w * u.labels)
    gap = sum((coef[(b, SEEN)] * total(s) + coef[(b, UNSEEN)] * total(u)
               for b, (s, u) in kept.items()), Fraction(0))
    k = 1
    for c in coef.values():
        k = math.lcm(k, c.denominator)
    n_items = sum(len(s.by_question) + len(u.by_question) for s, u in kept.values())
    strata = {}
    for b, (s, u) in kept.items():
        for name, part in ((SEEN, s), (UNSEEN, u)):
            a = n_items * int(coef[(b, name)] * k)
            xs = [x for _, x in part.by_question.values()]
            strata[f"{b}:{name}"] = Stratum(len(xs), a * sum(xs), a * a * sum(x * x for x in xs))
    bound = bound_from_strata(strata, margin_units=MEMO_MARGIN_UNITS * k, n_min=0,
                              n0=MEMO_MIN_LABELS)
    lower = None if bound.lower is None else bound.lower // k
    refused = n_s >= MEMO_MIN_LABELS and n_u >= MEMO_MIN_LABELS and bound.passes
    return MemoEvidence(MappingProxyType(kept), n_s, n_u, gap, lower, bound, refused, left_out)


def reference_verdicts(memo, window, sides, *, now):
    if memo.index.epoch != int(now):
        raise ValueError(f"the memorisation input is for epoch {memo.index.epoch}, not {now}")
    ref = labels_v4.reference(window, now=now)
    return {h: reference_evidence_from_parts(
                reference_split_labels(window, ref, sides[h], memo.published, now=now))
            for h in sorted(sides)}


# --- comparing --------------------------------------------------------------------------------


def plain(parts) -> dict:
    """The split as plain Python values, types included."""
    out = {}
    for b, p in parts.items():
        out[b] = {}
        for name, part in p.items():
            items = list(part.by_question.items())
            assert all(type(c) is int and type(s) is int for _, (c, s) in items)
            assert type(part.labels) is int
            out[b][name] = (part.labels, items, part.questions, part.total)
    return out


def same_split(window, ref, side, published, now, w=None):
    got = memo_v4.split_labels(window, ref, side, published, now=now, window_epochs=w)
    want = reference_split_labels(window, ref, side, published, now=now, window_epochs=w)
    assert list(got) == list(want)
    assert got == want and want == got
    assert plain(got) == plain(want)
    for b in got:
        for name in (SEEN, UNSEEN):
            assert got[b][name].by_question == want[b][name].by_question
            assert want[b][name].by_question == got[b][name].by_question
            assert dict(got[b][name].by_question) == dict(want[b][name].by_question)
    ev, want_ev = memo_v4.evidence_from_parts(got), reference_evidence_from_parts(want)
    assert ev == want_ev
    assert (ev.gap, ev.lower, ev.bound, ev.refused, ev.left_out, ev.seen_labels,
            ev.unseen_labels) == (want_ev.gap, want_ev.lower, want_ev.bound, want_ev.refused,
                                  want_ev.left_out, want_ev.seen_labels, want_ev.unseen_labels)
    assert type(ev.gap) is Fraction and ev.bound == want_ev.bound
    assert memo_v4.evidence_for_reveal(ev) == memo_v4.evidence_for_reveal(want_ev)
    assert memo_v4.memo_evidence(window, ref, side, published, now=now, window_epochs=w) == want_ev
    return got, ev


def test_the_exact_sums_are_python_int_sums_at_the_extremes():
    rng = np.random.default_rng(11)
    lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    cases = [np.zeros(0, np.int64), np.zeros(5, np.int64), np.array([lo], np.int64),
             np.array([hi, hi, hi], np.int64), np.array([lo, hi, -1, 0, 1], np.int64),
             np.array([lo] * 7 + [hi] * 3, np.int64)]
    for bits in (1, 8, 20, 31, 32, 40, 50, 62, 63):
        for n in (1, 3, 1000, 70_000):
            top = 2 ** min(bits, 63) - 1
            cases.append(rng.integers(-top - 1 if bits == 63 else -top, top, n,
                                      dtype=np.int64, endpoint=True))
    for a in cases:
        xs = [int(x) for x in a.tolist()]
        assert int_sum(a) == sum(xs)
        assert int_sum_squares(a) == sum(x * x for x in xs)
        o = np.array(xs + [2**70], dtype=object)
        assert int_sum(o) == sum(xs) + 2**70
        assert int_sum_squares(o) == sum(x * x for x in xs) + 2**140


# --- the memo tests' windows ---------------------------------------------------------------------


def test_the_memo_tests_windows_split_identically():
    from tests.test_memo_v4 import _window

    w = _window(range(10, 20))
    w.odds["h"] = {e: np.full(40, 6000, np.int16) for e in range(10, 20)}
    ref = labels_v4.reference(w, now=19)
    pub = Published({f"e{e}q{i}": e for e in range(10, 20) for i in range(40)}, 19)
    repeat = Published({f"e{e}q{i}": 3 if e >= 15 else e
                        for e in range(10, 20) for i in range(40)}, 19)
    for p in (pub, repeat):
        for side in (Side("h", "ckA", 14), Side("h", "ckZ", 12),
                     Side("h", "ckZ", 12, rows_since={"m/a": 2}),
                     Side("h", "ckZ", 12, rows_since={"m/a": 2, "m/b": 2}),
                     Side("h", "ckB", 25), Side("h", "ckZ", -10**30, rows_since={"m/b": 10**40})):
            same_split(w, ref, side, p, 19)
            same_split(w, ref, side, p, 19, w=4)
    for wrong in ((Side("h", "ckZ", 12), Published({}, 17), "folded through"),
                  (labels_v4.REFERENCE, pub, "not of R")):
        for fn in (memo_v4.split_labels, reference_split_labels):
            with pytest.raises(ValueError, match=wrong[2]):
                fn(w, ref, wrong[0], wrong[1], now=19)


def test_an_epoch_that_is_not_v4_raises_only_where_the_head_has_odds():
    from tests.test_memo_v4 import _window

    w = _window(range(10, 14))
    plain_pairs = w.pairs[11]
    fields = {f: getattr(plain_pairs, f) for f in ("question_ids", "models", "benchmarks",
                                                   "categories", "y", "prompt_tokens",
                                                   "completion_tokens")}
    w.pairs[11] = EpochPairs(**fields)
    w.odds["h"] = {e: np.full(40, 6000, np.int16) for e in (10, 12, 13)}
    ref = labels_v4.reference(w, now=13)
    pub = Published({}, 13)
    same_split(w, ref, Side("h", "ckZ", 0), pub, 13)          # no odds on 11: skipped
    w.odds["h"][11] = np.full(40, 6000, np.int16)
    for fn in (memo_v4.split_labels, reference_split_labels):
        with pytest.raises(ValueError, match="not a v4 epoch"):
            fn(w, ref, Side("h", "ckZ", 0), pub, now=13)


def test_the_published_44_epoch_reproduction_splits_identically():
    from tests.memo_v4_observed import NOW, load

    window, published, sides = load()
    ref = labels_v4.reference(window, now=NOW)
    split = memo_v4._Split(window, ref, published, now=NOW)
    for h, side in sorted(sides.items()):
        want = reference_split_labels(window, ref, side, published, now=NOW)
        got = split.split(side)
        assert got == want and plain(got) == plain(want)
        assert memo_v4.evidence_from_parts(got) == reference_evidence_from_parts(want)
    # and the test's own pinned figures, through the module
    for h, side in list(sorted(sides.items()))[:5]:
        same_split(window, ref, side, published, NOW)


def test_the_seat_memo_world_verdicts_and_reveal_are_identical():
    from fugal_subnet import seat_v4
    from fugal_subnet.attacks import memo_seat_world as msw

    world = msw.World()
    for kept in (("honest", "mem"), ("honest", "line1", "mem"), ("mem",)):
        window = world.window(kept)
        memo = world.memo()
        state = world.seat(None, kept)
        sides = {hs.head: seat_v4.side_of(hs) for hs in state.heads}
        got = memo_v4.verdicts(memo, window, sides, now=msw.NOW)
        want = reference_verdicts(memo, window, sides, now=msw.NOW)
        assert got == want
        assert any(ev.refused for ev in got.values())
        a = json.dumps(memo_v4.memo_for_reveal(memo, got, now=msw.NOW), sort_keys=True)
        b = json.dumps(memo_v4.memo_for_reveal(memo, want, now=msw.NOW), sort_keys=True)
        assert a == b


# --- randomised windows ------------------------------------------------------------------------


def _random_world(rng, k):
    """A window of random LabelPairs: shared questions across epochs (so a
    question collects labels from several epochs and entries), entries some
    heads lack, coldkeys that own heads voting, weights or none, and heads
    with odds on some epochs only."""
    n_epochs = int(rng.integers(1, 9))
    now = 100 + int(rng.integers(0, 3))
    epochs = sorted(rng.choice(np.arange(now - 12, now + 1), n_epochs, replace=False).tolist())
    entries = [f"m/{i}" for i in range(int(rng.integers(1, 6)))]
    benches = [f"b{i}" for i in range(int(rng.integers(1, 5)))]
    coldkeys = [f"ck{i}" for i in range(int(rng.integers(1, 6)))]
    n_q = int(rng.integers(3, 60))
    q_bench = {f"q{i:03d}": benches[int(rng.integers(0, len(benches)))] for i in range(n_q)}
    qs = sorted(q_bench)
    weighted = bool(rng.integers(0, 2))
    huge = k % 23 == 0
    pairs = {}
    for e in epochs:
        n = int(rng.integers(0, 80))
        cells = sorted({(qs[int(rng.integers(0, n_q))], entries[int(rng.integers(0, len(entries)))])
                        for _ in range(n)})
        n = len(cells)
        voters = [tuple(sorted(set(rng.choice(coldkeys, int(rng.integers(1, 3))).tolist())))
                  for _ in range(n)]
        y = rng.integers(0, 2, n).astype(np.int32) * 10_000
        if k % 7 == 0:
            y = rng.integers(0, 10_001, n).astype(np.int32)
        if huge:      # skill sums past int64: the sums are of Python ints
            y = rng.choice(np.array([-2_100_000_000, 2_100_000_000], np.int32), n)
        weight = None
        if weighted and not huge:
            weight = rng.integers(1, 3_000_000, n).astype(np.int64)
        base = EpochPairs(question_ids=tuple(c[0] for c in cells),
                          models=tuple(c[1] for c in cells),
                          benchmarks=tuple(q_bench[c[0]] for c in cells),
                          categories=tuple("cat-" + q_bench[c[0]] for c in cells),
                          y=y, prompt_tokens=np.zeros(n, np.int32),
                          completion_tokens=np.zeros(n, np.int32), weight=weight)
        missing = rng.choice(np.array([-1, 5000, 3000], np.int16), n)
        pairs[e] = labels_v4.label_pairs(base, voters, missing)
    heads = {}
    for j in range(int(rng.integers(1, 6))):
        h = f"h{j}"
        heads[h] = {e: rng.integers(-1, 10_001, len(pairs[e].question_ids)).astype(np.int16)
                    for e in epochs if rng.random() < 0.8}
    window = Window(size=48, slices={}, heads={}, pairs=pairs, odds=heads)
    published_through = now - 1
    first = {}
    for q in qs:
        r = rng.random()
        if r < 0.6:
            first[q] = int(rng.integers(now - 20, published_through + 1))
    published = Published(first, published_through)
    sides = {}
    for h in heads:
        since = int(rng.integers(now - 15, now + 2))
        rows = {}
        if rng.random() < 0.5:                       # a successor: inherited rows
            for m in entries:
                if rng.random() < 0.6:
                    rows[m] = int(rng.integers(now - 30, since + 1))
        owner = coldkeys[int(rng.integers(0, len(coldkeys)))] if rng.random() < 0.7 else "ck-none"
        sides[h] = Side(h, owner, since, rows)
    return window, published, sides, now


def test_hundreds_of_random_windows_split_identically():
    rng = np.random.default_rng(20261120)
    seen_kinds = {"left_out": 0, "empty": 0, "successor": 0, "one_side": 0,
                  "object": 0}
    for k in range(300):
        window, published, sides, now = _random_world(rng, k)
        ref = labels_v4.reference(window, now=now)
        w = None if k % 5 else int(rng.integers(1, 10))
        split = memo_v4._Split(window, ref, published, now=now, window_epochs=w)
        for h, side in sorted(sides.items()):
            got = split.split(side)
            want = reference_split_labels(window, ref, side, published, now=now, window_epochs=w)
            assert got == want and plain(got) == plain(want), (k, h)
            ev = memo_v4.evidence_from_parts(got)
            assert ev == reference_evidence_from_parts(want), (k, h)
            seen_kinds["left_out"] += bool(ev.left_out)
            seen_kinds["empty"] += not got
            seen_kinds["successor"] += bool(side.rows_since)
            seen_kinds["one_side"] += any(not p[SEEN].labels or not p[UNSEEN].labels
                                          for p in got.values())
            seen_kinds["object"] += any(p[s].by_question._sums.dtype == object
                                        for p in got.values() for s in (SEEN, UNSEEN))
    assert all(seen_kinds.values()), seen_kinds


def test_refusals_on_random_windows_match_the_reference():
    """Windows built so heads are refused or not: a memorising head (skill on
    seen questions only), an honest one and one at the margin, many
    questions, through verdicts and the reveal."""
    from fugal_subnet import epoch_plan
    from fugal_subnet.memo_v4 import EpochMemo

    rng = np.random.default_rng(7)
    refusals, reserves = [], set()
    sets = sorted(epoch_plan.mix_v4().bench_category)
    pool = [{"question_id": f"{s}_{i:03d}", "benchmark": s, "prompt": "x"}
            for s in sets for i in range(30)]
    for trial in range(6):
        now = 60
        pairs, first, odds = {}, {}, {"mem": {}, "honest": {}, "edge": {}}
        for e in range(now - 20, now):
            q = [f"o{e}_{i}" for i in range(150)] + [f"f{e}_{j}" for j in range(8) for _ in range(5)]
            for i in range(150):
                first[f"o{e}_{i}"] = 5
            for j in range(8):
                first[f"f{e}_{j}"] = e
            n = len(q)
            m = tuple(f"m/{(i * 7) % 4}" for i in range(n))
            b = tuple("b-easy" if i % 2 else "b-hard" for i in range(n))
            y = rng.integers(0, 2, n).astype(np.int32) * 10_000
            base = EpochPairs(question_ids=tuple(q), models=m, benchmarks=b,
                              categories=("knowledge",) * n, y=y,
                              prompt_tokens=np.zeros(n, np.int32),
                              completion_tokens=np.zeros(n, np.int32),
                              weight=rng.integers(500_000, 2_000_000, n).astype(np.int64))
            pairs[e] = labels_v4.label_pairs(base, [(f"ck{i % 5}",) for i in range(n)],
                                             np.full(n, -1, np.int16))
            fresh = np.array([x.startswith("f") for x in q])
            truth = np.where(y > 0, 9000, 1000)
            base_odds = np.full(n, 5000)
            odds["mem"][e] = np.where(~fresh & (rng.random(n) < 0.6 + 0.05 * trial), truth,
                                      base_odds).astype(np.int16)
            odds["honest"][e] = np.where(rng.random(n) < 0.4, truth, base_odds).astype(np.int16)
            odds["edge"][e] = np.where(rng.random(n) < (0.45 if not trial % 2 else 0.4)
                                       + 0.1 * ~fresh, truth, base_odds).astype(np.int16)
        window = Window(size=48, slices={}, heads={}, pairs=pairs, odds=odds)
        published = Published(first, now - 1)
        index = memo_v4.reserve_index(Published({q["question_id"]: 1 for q in pool}
                                                if trial % 2 else {}, now - 2), pool, now)
        memo = EpochMemo(index, (), published)
        sides = {h: Side(h, "ck-owner", 10, {"m/1": 2} if h == "edge" else {}) for h in odds}
        got = memo_v4.verdicts(memo, window, sides, now=now)
        want = reference_verdicts(memo, window, sides, now=now)
        assert got == want
        assert got["mem"].refused and not got["honest"].refused
        refusals.append(got["edge"].refused)
        reserves.add(index.reserve_size)
        a = json.dumps(memo_v4.memo_for_reveal(memo, got, now=now), sort_keys=True)
        assert a == json.dumps(memo_v4.memo_for_reveal(memo, want, now=now), sort_keys=True)
    assert 0 in reserves and len(reserves) == 2           # an empty reserve and a full one


def test_a_split_built_part_reads_as_the_mapping_it_replaces():
    from tests.test_memo_v4 import _window

    w = _window(range(10, 14))
    w.odds["h"] = {e: np.full(40, 6000, np.int16) for e in range(10, 14)}
    ref = labels_v4.reference(w, now=13)
    got = memo_v4.split_labels(w, ref, Side("h", "ckZ", 0), Published({}, 13), now=13)
    want = reference_split_labels(w, ref, Side("h", "ckZ", 0), Published({}, 13), now=13)
    for b in got:
        g, r = got[b][UNSEEN].by_question, want[b][UNSEEN].by_question
        assert list(g) == list(r) and list(g.values()) == list(r.values())
        assert list(g.items()) == list(r.items()) and len(g) == len(r)
        q = next(iter(r))
        assert g[q] == r[q] and q in g and g.get("nope") is None
        assert repr(g) == repr(r)
        with pytest.raises(TypeError):
            hash(g)
        with pytest.raises(TypeError):
            g["x"] = (1, 2)
