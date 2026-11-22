"""Which labels a comparison reads (labels_v4.eligible) and the sums rule A
and the published standing read (labels_v4.gain_strata), computed per entry
and per owner and summed with integer arrays, are exactly what the per-label
loops they replaced gave.

The loops are kept here as the reference (labels_v4 at 923170e, verbatim but
for names) and compared on the published 44-epoch reproduction, the seat's
synthetic world through twenty epochs, and a few hundred randomised windows
(inherited rows, owners voting, missing odds, R on either side, sums past
int64), dict order included."""
from __future__ import annotations

import gc

import numpy as np

from fugal_subnet import labels_v4
from fugal_subnet.labels_v4 import REFERENCE, Side
from fugal_subnet.seat_bound import strata
from tests.test_memo_v4_vectorised import _random_world, reference_eligible


def reference_gain_items(window, ref, x, y, *, now, window_epochs=labels_v4.SEAT_WINDOW_EPOCHS):
    from fugal_subnet.explore_weights import apply

    out: dict[str, list[int]] = {}
    for e in sorted(window.pairs):
        if not labels_v4.in_window(e, now, window_epochs):
            continue
        pairs = window.pairs[e]
        ox, oy = labels_v4._scored(window, ref, x, e), labels_v4._scored(window, ref, y, e)
        if ox is None or oy is None:
            continue
        keep = reference_eligible(pairs, e, x, y, now=now, window_epochs=window_epochs)
        if not keep.any():
            continue
        t = pairs.y.astype(np.int64)
        units = (oy - t) ** 2 - (ox - t) ** 2
        if pairs.weight is not None:
            units = apply(units, pairs.weight)
        for c, u in zip(np.asarray(pairs.categories)[keep].tolist(), units[keep].tolist()):
            out.setdefault(str(c), []).append(int(u))
    return out


def reference_gain_strata(window, ref, x, y, *, now, window_epochs=labels_v4.SEAT_WINDOW_EPOCHS):
    return strata(reference_gain_items(window, ref, x, y, now=now, window_epochs=window_epochs))


def same(window, ref, x, y, now, w=labels_v4.SEAT_WINDOW_EPOCHS):
    for e in sorted(window.pairs):
        p = window.pairs[e]
        if isinstance(p, labels_v4.LabelPairs):
            got = labels_v4.eligible(p, e, x, y, now=now, window_epochs=w)
            want = reference_eligible(p, e, x, y, now=now, window_epochs=w)
            assert got.dtype == want.dtype and got.shape == want.shape
            assert np.array_equal(got, want)
    got = labels_v4.gain_strata(window, ref, x, y, now=now, window_epochs=w)
    want = reference_gain_strata(window, ref, x, y, now=now, window_epochs=w)
    assert list(got.items()) == list(want.items())
    assert all(type(v) is int for s in got.values() for v in (s.n, s.total, s.squares))
    items = labels_v4.gain_items(window, ref, x, y, now=now, window_epochs=w)
    assert items == reference_gain_items(window, ref, x, y, now=now, window_epochs=w)
    assert list(items) == list(want)
    return got


def test_random_windows_read_the_same_labels_and_sums():
    rng = np.random.default_rng(31)
    shown = {"empty": 0, "huge": 0, "several_categories": 0}
    for k in range(250):
        window, _published, sides, now = _random_world(rng, k)
        ref = labels_v4.reference(window, now=now)
        w = labels_v4.SEAT_WINDOW_EPOCHS if k % 4 else int(rng.integers(1, 10))
        names = sorted(sides)
        for i, h in enumerate(names):
            x = sides[h]
            for y in [REFERENCE] + [sides[g] for g in names[i + 1:]]:
                for a, b in ((x, y), (y, x)):
                    got = same(window, ref, a, b, now, w)
                    shown["empty"] += not got
                    shown["several_categories"] += len(got) > 1
                    shown["huge"] += any(abs(s.total) >= 2**63 or s.squares >= 2**63
                                         for s in got.values())
    assert all(shown.values()), shown


def test_the_published_epochs_compare_heads_identically():
    from tests.memo_v4_observed import NOW, load

    window, _published, sides = load()
    ref = labels_v4.reference(window, now=NOW)
    names = sorted(sides)[:12]
    for i, h in enumerate(names):
        same(window, ref, sides[h], REFERENCE, NOW)
        for g in names[i + 1:i + 3]:
            same(window, ref, sides[h], sides[g], NOW)
            same(window, ref, sides[g], Side(h, sides[h].owner, sides[h].since,
                                             {"x": 3}), NOW)


def test_the_seat_world_compares_identically_through_twenty_epochs():
    from tests.seat_v4_world import World

    w = World(labels=300, seed=4)
    w.add("a", skill=0.3, block=3)
    w.add("b", skill=0.5, block=4, noise=2)
    w.add("c", skill=0.5, block=6, row_skill={"lab-b/model@low": 0.7})
    w.missing_after["lab-b/model@low"] = 9
    for e in range(1, 21):
        w.epoch(e, served=["a", "b", "c"] if e % 5 else ["a"], answered=["a", "b"])
        ref = labels_v4.reference(w.window, now=e)
        sides = [Side(h.hash, h.owner, 1, {"lab-a/model@high": 0} if h.name == "c" else {})
                 for h in w.heads.values()]
        for x in sides:
            same(w.window, ref, x, REFERENCE, e)
            for y in sides:
                if y is not x:
                    same(w.window, ref, x, y, e)


def test_the_label_index_lives_and_dies_with_its_labels():
    from fugal_subnet.champion import EpochPairs

    def labels(n, tag):
        base = EpochPairs(question_ids=tuple(f"{tag}{i}" for i in range(n)),
                          models=("m/a",) * n, benchmarks=("b",) * n, categories=("c",) * n,
                          y=np.zeros(n, np.int32), prompt_tokens=np.zeros(n, np.int32),
                          completion_tokens=np.zeros(n, np.int32))
        return labels_v4.label_pairs(base, [("ck",)] * n, np.full(n, -1, np.int16))

    before = len(labels_v4._INDEXES)
    for k in range(50):
        p = labels(5 + k, f"t{k}")
        assert labels_v4._label_index(p) is labels_v4._label_index(p)
        assert len(labels_v4._label_index(p).entry) == 5 + k
        del p
    gc.collect()
    assert len(labels_v4._INDEXES) <= before + 1
