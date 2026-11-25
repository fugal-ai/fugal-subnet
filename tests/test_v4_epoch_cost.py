"""The v4 epoch's cost stays where it was measured (scripts/experiments/
time_v4_epoch.py): the memorisation check and rule A's comparisons cost each
head a fixed number of Python steps per epoch and entry, however many labels
an epoch holds, so a per-label Python loop coming back is caught here rather
than at 256 UIDs, where it cost about an hour of a 72-minute epoch.

The quick tests count executed Python lines (sys.settrace over the v4 seat
modules), not seconds: per head, the lines run for a window with four times
the labels must not grow. The slow test times the epoch at full size with a
generous bound."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fugal_subnet import labels_v4, memo_v4
from fugal_subnet.champion import EpochPairs, Window
from fugal_subnet.labels_v4 import REFERENCE, Side
from fugal_subnet.memo_v4 import EpochMemo, Published

_MODULES = ("memo_v4.py", "labels_v4.py", "seat_bound.py", "seat_v4.py", "explore_weights.py")


def count_lines(fn):
    """(result, Python lines executed in the v4 seat modules while `fn` runs)."""
    n = [0]

    def tracer(frame, event, arg):
        if not frame.f_code.co_filename.endswith(_MODULES):
            return None

        def local(frame, event, arg):
            if event == "line":
                n[0] += 1
            return local
        return local

    sys.settrace(tracer)
    try:
        out = fn()
    finally:
        sys.settrace(None)
    return out, n[0]


def world(per_epoch: int, heads: int, epochs: int = 6):
    """A window of `epochs` epochs of `per_epoch` labels on four entries and
    two benchmarks, every head with odds on every epoch, and the memo's
    published set (half the questions published before the heads' rows)."""
    rng = np.random.default_rng(per_epoch)
    now = 50
    pairs, first, odds = {}, {}, {}
    for e in range(now - epochs + 1, now + 1):
        q = [f"q{e}_{i}" for i in range(per_epoch)]
        for i, x in enumerate(q):
            if i % 2:
                first[x] = 3
        n = per_epoch
        base = EpochPairs(question_ids=tuple(q), models=tuple(f"m/{i % 4}" for i in range(n)),
                          benchmarks=tuple("b1" if i % 3 else "b2" for i in range(n)),
                          categories=tuple("c1" if i % 3 else "c2" for i in range(n)),
                          y=rng.integers(0, 2, n).astype(np.int32) * 10_000,
                          prompt_tokens=np.zeros(n, np.int32), completion_tokens=np.zeros(n, np.int32),
                          weight=rng.integers(500_000, 2_000_000, n).astype(np.int64))
        pairs[e] = labels_v4.label_pairs(base, [(f"ck{i % 7}",) for i in range(n)],
                                         np.full(n, -1, np.int16))
    sides = {}
    for j in range(heads):
        h = f"h{j:02d}"
        odds[h] = {e: rng.integers(0, 10_001, per_epoch).astype(np.int16) for e in pairs}
        sides[h] = Side(h, f"ck{j % 9}", 10, {"m/1": 2} if j % 2 else {})
    window = Window(size=48, slices={}, heads={}, pairs=pairs, odds=odds)
    return window, Published(first, now), sides, now


class _Index:
    epoch = 51


def memo_lines(per_epoch: int, heads: int) -> int:
    window, published, sides, now = world(per_epoch, heads)
    memo = EpochMemo(_Index(), (), published)
    memo.index.epoch = now
    _out, n = count_lines(lambda: memo_v4.verdicts(memo, window, sides, now=now))
    return n


def test_the_memorisation_check_costs_a_head_the_same_however_many_labels():
    """Per head (the lines of eight heads minus those of two, over six), the
    check runs no more Python for 2,000 labels an epoch than for 500: what
    grows with the labels is shared by every head, or is array work."""
    small = (memo_lines(500, 8) - memo_lines(500, 2)) / 6
    large = (memo_lines(2000, 8) - memo_lines(2000, 2)) / 6
    assert large <= small * 1.1 + 50, (small, large)
    # and the shared part grows no faster than the labels (one pass each)
    assert memo_lines(2000, 2) < 5 * memo_lines(500, 2)


def test_a_rule_a_comparison_costs_the_same_however_many_labels():
    """One gain_strata comparison (rule A and C, and the published standing)
    runs a fixed number of Python lines per epoch, entry and category: the
    labels are array work. The per-epoch index is built once per epoch's
    labels and reused by every comparison."""
    def lines(per_epoch):
        window, _published, sides, now = world(per_epoch, 3)
        ref = labels_v4.reference(window, now=now)
        x, y = sides["h01"], sides["h02"]
        labels_v4.gain_strata(window, ref, x, y, now=now)           # builds the index
        _out, a = count_lines(lambda: labels_v4.gain_strata(window, ref, x, y, now=now))
        _out, b = count_lines(lambda: labels_v4.gain_strata(window, ref, x, REFERENCE, now=now))
        return a, b

    (a1, b1), (a4, b4) = lines(500), lines(2000)
    assert a4 <= a1 * 1.1 + 20 and b4 <= b1 * 1.1 + 20, (a1, a4, b1, b4)


@pytest.mark.slow
# ~12 minutes each on a CI runner, so CI runs these in their own jobs on
# release/* pull requests, pushes to main and release/**, and nightly, not in
# every pull request's shards. The line-count tests above are the per-PR guard
# against the per-label loop this file exists for.
@pytest.mark.capacity
@pytest.mark.parametrize("shape", ["ladder", "served_last"])
def test_one_v4_epoch_at_full_size_fits_well_inside_the_epoch(shape):
    """scripts/experiments/time_v4_epoch.py at full size (256 UIDs, 48
    epochs, 768 kept heads, 64 entries, 1024-wide embeddings) under the
    adversarial ladder (rule C tests nearly every earlier contender), and
    with every served head a candidate against all 512 kept heads and every
    served head before it (`--served-last`, the worst shape measured), every
    served head but the holder a challenger of 1,000 questions whose picks
    and the holder's are recomputed: every stage within generous bounds, the
    whole run_v2_epoch within fifteen minutes, and the epoch's own memory
    under 4 GiB. Single process (the comparisons use a few threads).

    Memory is the growth of the resident set while the epoch runs, sampled,
    over what the process held when the test began: the process's high-water
    mark (ru_maxrss) cannot be reset and also holds whatever ran earlier in
    the same process, so in a whole-suite run it measures the suite."""
    import gc
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "experiments",
                        "time_v4_epoch.py")
    spec = importlib.util.spec_from_file_location("time_v4_epoch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # the dataclasses resolve their module through it
    spec.loader.exec_module(mod)
    gc.collect()
    before = mod.rss_mib()
    with mod.Sampler() as memory:
        w = mod.EpochWorld(mod.Shape(served_last=shape == "served_last"))
        out = mod.run_stages(w, whole=True, report=lambda *_a: None)
    bounds = {"pool + current odds": 180, "add epoch + late odds": 120,
              "memorisation evidence (all kept)": 180, "standing over R (serving heads)": 120,
              "seat observe": 30, "challenges (both picks recomputed)": 180,
              "holder picks on the list (published)": 30,
              "seat decide (rules A, B, C)": 600, "build the reveal (_v4_result)": 120,
              "save the window": 60, "load the window": 60,
              "WHOLE run_v2_epoch (with memo)": 900}
    slow = {k: round(out[k][0]) for k, b in bounds.items() if out[k][0] > b}
    assert not slow, slow
    assert out["_same_reveal"]
    assert len(w.verified) - 1 == sum(1 for p in w.verified.values() if p.scored)
    assert memory.peak - before < 4 * 1024, (round(before), round(memory.peak))
