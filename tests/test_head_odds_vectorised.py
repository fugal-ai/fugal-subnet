"""The vectorised head odds (head_odds.odds_matrix, odds_grid, odds_by_row,
pair_odds_v4_many; labels_v4.odds_on_many) give exactly the integers of
the exact path, head_odds.odds (an fsum of exact products, compared with the
pinned threshold table), and fall back to it inside the error band.

The exact path is the reference, called here per (row, question). Compared:
a few hundred thousand random cells with a fixed seed (trained-scale and
extreme heads, unit and non-unit embeddings), heads at the size limits
(|W|, |b| up to 1e6) including rows built to cancel, logits placed exactly
on and next to thresholds (the band must send those to the exact path), the
v4 callers against their per-pair forms, and the real pinned embeddings when
the file is on this machine (FUGAL_EMBEDDINGS_FILE with the pinned hash;
skipped otherwise: no network in tests)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from fugal_subnet import head_odds


def exact_matrix(W, b, rows, H):
    return np.array([[head_odds.odds(W[r], b[r], H[q]) for r in rows] for q in range(len(H))],
                    dtype=np.int16).reshape(len(H), len(rows))


def unit_rows(rng, n, d, dtype=np.float32):
    h = rng.standard_normal((n, d))
    return (h / np.linalg.norm(h, axis=1, keepdims=True)).astype(dtype)


class Counting:
    """Counts the exact path's calls made by the vectorised functions."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = head_odds.odds

        def counted(*a):
            self.calls += 1
            return real(*a)
        monkeypatch.setattr(head_odds, "odds", counted)


def test_random_cells_equal_the_exact_path():
    """200,000 cells at the v4 width (1,024) and 120,000 at small widths,
    heads from trained scale (|W| ~ 0.05) to the 1e6 limit, float32 and
    float64 parameters, unit and non-unit embeddings."""
    rng = np.random.default_rng(20261121)
    cells = 0
    for k in range(40):
        d = 1024 if k < 20 else int(rng.integers(1, 40))
        rows = int(rng.integers(1, 12))
        q = (200_000 // 20 // rows) if k < 20 else (120_000 // 20 // rows)
        scale = [0.05, 1.0, 1e-6, 3e3, 1e6][k % 5]
        W = rng.standard_normal((rows, d)) * scale
        b = rng.standard_normal(rows) * [0.5, 5, 1e-6, 1e3, 1e6][(k // 5) % 5]
        W = np.clip(W, -1e6, 1e6)
        b = np.clip(b, -1e6, 1e6)
        if k % 2:
            W, b = W.astype(np.float32), b.astype(np.float32)
        H = unit_rows(rng, q, d) if k % 3 else rng.uniform(-1.001, 1.001, (q, d)).astype(np.float32)
        got = head_odds.odds_matrix(W, b, list(range(rows)), H)
        want = exact_matrix(W, b, list(range(rows)), H)
        assert got.dtype == np.int16 and np.array_equal(got, want), k
        cells += got.size
    assert cells >= 300_000


def test_logits_on_and_beside_thresholds_go_to_the_exact_path(monkeypatch):
    """A row of zeros with the bias exactly a threshold (or one ulp either
    side), and a row whose product with a basis embedding is a threshold:
    the float result is within the band of a threshold, so the exact path
    decides, and gives what it gives on its own."""
    t = head_odds.thresholds()
    d = 1024
    picks = [0, 1, 17, 4999, 5000, 9998, 9999]
    biases, rows = [], []
    for k in picks:
        for v in (t[k], np.nextafter(t[k], np.inf), np.nextafter(t[k], -np.inf)):
            biases.append(v)
            rows.append(np.zeros(d))
    W = np.array(rows)
    b = np.array(biases)
    H = unit_rows(np.random.default_rng(1), 3, d)
    count = Counting(monkeypatch)
    got = head_odds.odds_matrix(W, b, list(range(len(b))), H)
    assert count.calls >= len(b) * len(H) - 0         # every one is inside the band
    count.calls = 0
    want = exact_matrix(W, b, list(range(len(b))), H)
    assert np.array_equal(got, want)
    # on a threshold: bisect_right counts it
    assert int(got[0, 0]) == head_odds.odds_of_logit(t[0])

    # the product itself lands on a threshold
    e1 = np.zeros((1, d), np.float32)
    e1[0, 7] = 1.0
    W2 = np.zeros((len(picks), d))
    W2[:, 7] = [t[k] for k in picks]
    count.calls = 0
    got2 = head_odds.odds_matrix(W2, np.zeros(len(picks)), list(range(len(picks))), e1)
    assert count.calls == len(picks)
    assert got2[0].tolist() == [k + 1 for k in picks]


def test_heads_at_the_size_limits_equal_the_exact_path(monkeypatch):
    """|W| and |b| at 1e6: rows of full magnitude (logits far past the
    table: settled without the exact path), and rows built to cancel on
    their questions (w = 1e6 times a vector orthogonal to h, plus a small
    bias), whose logits sit among the thresholds with a band of about 1e-4:
    some go to the exact path, and every integer is the exact one."""
    rng = np.random.default_rng(7)
    d = 1024
    H = unit_rows(rng, 400, d)
    W = np.sign(rng.standard_normal((6, d))) * 1e6
    b = np.array([1e6, -1e6, 0.0, 5e5, -3.0, 1e6])
    count = Counting(monkeypatch)
    got = head_odds.odds_matrix(W, b, list(range(6)), H)
    assert np.array_equal(got, exact_matrix(W, b, list(range(6)), H))

    rows, biases, qs = [], [], []
    for i in range(300):
        h = H[i].astype(np.float64)
        r = rng.standard_normal(d)
        w = r - (r @ h) * h
        w = w / np.abs(w).max() * 1e6
        rows.append(w)
        biases.append(float(rng.uniform(-6, 6)))
        qs.append(i)
    W2 = np.array(rows)
    b2 = np.array(biases)
    count.calls = 0
    got2 = head_odds.odds_by_row(W2, b2, list(range(300)), H[qs])
    fell_back = count.calls
    want2 = np.array([head_odds.odds(W2[i], b2[i], H[qs[i]]) for i in range(300)], np.int16)
    assert np.array_equal(got2, want2)
    assert fell_back > 0


def test_the_v4_callers_equal_their_per_pair_forms(monkeypatch):
    """pair_odds_v4_many against pair_odds_v4 per head, odds_by_row against
    a loop of head_odds.odds, picks_and_odds_v4's odds against odds, and
    labels_v4.odds_on_many against odds_on's former per-pair form
    (head_odds.head_odds), on a menu with entries some heads lack."""
    from fugal_subnet import labels_v4
    from fugal_subnet.attacks import memo_seat_world as msw
    from fugal_subnet.champion import EpochPairs

    rng = np.random.default_rng(11)
    d = 1024
    entries = tuple(f"lab-{i}/m@high" for i in range(6))
    state = msw.menu(entries=entries, epoch=40)
    ids = state.ids()
    heads = []
    for k in range(5):
        models = [m for m in ids if rng.random() < 0.8] + ["off/menu@x"]
        rng.shuffle(models)
        W = (rng.standard_normal((len(models), d)) * 0.05).astype(np.float32)
        b = (rng.standard_normal(len(models)) * 0.3).astype(np.float32)
        heads.append(SimpleNamespace(W=W, b=b, models=models))
    n = 700
    H = unit_rows(rng, n, d)
    pair_entries = [ids[int(i)] for i in rng.integers(0, len(ids), n)]
    many = head_odds.pair_odds_v4_many(heads, state, list(H), pair_entries)
    for head, got in zip(heads, many):
        rows = head_odds.menu_rows(head.models, ids)
        row_of = dict(zip(ids, rows))
        want = np.array([-1 if row_of[e] is None else head_odds.odds(head.W[row_of[e]],
                                                                   head.b[row_of[e]], H[i])
                         for i, e in enumerate(pair_entries)], dtype=np.int16)
        assert np.array_equal(got, want)
        assert np.array_equal(head_odds.pair_odds_v4(head, state, list(H), pair_entries), want)
    with pytest.raises(ValueError, match="not on the menu"):
        head_odds.pair_odds_v4_many(heads, state, list(H[:2]), [ids[0], "nope"])

    mo = head_odds.picks_and_odds_v4(heads[0], state, H[:50])
    rows = head_odds.menu_rows(heads[0].models, ids)
    for q in range(50):
        for j, r in enumerate(rows):
            want = -1 if r is None else head_odds.odds(heads[0].W[r], heads[0].b[r], H[q])
            assert int(mo.odds[q, j]) == want

    base = EpochPairs(question_ids=tuple(f"q{i}" for i in range(n)), models=tuple(pair_entries),
                      benchmarks=("b",) * n, categories=("c",) * n,
                      y=np.zeros(n, np.int32), prompt_tokens=np.zeros(n, np.int32),
                      completion_tokens=np.zeros(n, np.int32))
    pairs = labels_v4.label_pairs(base, [("ck",)] * n, rng.choice([-1, 4000], n))
    hidden_of = {f"q{i}": H[i] for i in range(n)}.__getitem__
    triples = [(h.W, h.b, h.models) for h in heads]
    for (W, b, models), got in zip(triples, labels_v4.odds_on_many(triples, pairs, hidden_of)):
        old = head_odds.head_odds(W, b, list(models), [hidden_of(q) for q in pairs.question_ids],
                                  list(pairs.models))
        assert np.array_equal(got, labels_v4.effective(old, pairs))
        assert np.array_equal(labels_v4.odds_on((W, b, models), pairs, hidden_of), got)


def test_ragged_or_misaligned_embeddings_raise_as_the_exact_path_does():
    W = np.zeros((2, 4))
    b = np.zeros(2)
    with pytest.raises(ValueError, match="do not align"):
        head_odds.odds_by_row(W, b, [0, 1], [np.zeros(4), np.zeros(5)])
    with pytest.raises(ValueError, match="do not align"):
        head_odds.odds_by_row(W, b, [0], np.zeros((1, 3)))
    assert head_odds.odds_by_row(W, b, [-1, -1], np.zeros((2, 4))).tolist() == [-1, -1]
    assert head_odds.odds_matrix(W, b, [], np.zeros((3, 4))).shape == (3, 0)


def _pinned_embeddings():
    from fugal_subnet import config
    from fugal_subnet.embeddings_file import Embeddings

    path = os.environ.get("FUGAL_EMBEDDINGS_FILE")
    if not path or not os.path.exists(path) or os.path.getsize(path) < 100_000_000:
        pytest.skip("the pinned embeddings file is not on this machine (FUGAL_EMBEDDINGS_FILE)")
    try:
        return Embeddings.load(path, config.EMBEDDINGS_SHA256)
    except ValueError:
        pytest.skip("FUGAL_EMBEDDINGS_FILE is not the pinned embeddings file")


def test_the_real_pinned_embeddings_give_the_exact_integers():
    """3,000 pool questions of the pinned file (spread over it) against
    trained-scale heads, a sharp head and a head at the size limit."""
    emb = _pinned_embeddings()
    rng = np.random.default_rng(5)
    H = emb.hidden[np.linspace(0, len(emb.hidden) - 1, 3000).astype(int)]
    d = H.shape[1]
    for scale, bscale, rows in ((0.05, 0.5, 24), (2.0, 3.0, 8), (1e6, 1e6, 4)):
        W = (rng.standard_normal((rows, d)) * scale).astype(np.float32)
        b = (rng.standard_normal(rows) * bscale).astype(np.float32)
        W, b = np.clip(W, -1e6, 1e6), np.clip(b, -1e6, 1e6)
        got = head_odds.odds_matrix(W, b, list(range(rows)), H)
        assert np.array_equal(got, exact_matrix(W, b, list(range(rows)), H)), scale


def test_the_vectorised_pick_is_the_fraction_pick():
    """head_odds.pick_rows against the per-question Fraction loop it
    replaced: random odds, prices as exact Fractions with large and small
    denominators, entries without rows, and ties built on purpose (equal
    prices and odds, and utilities equal across different odds and prices),
    which the exact comparison decides, to the earlier entry."""
    from fractions import Fraction

    from fugal_subnet.head_odds import ODDS_SCALE, _exact_pick, pick_rows

    rng = np.random.default_rng(23)
    ties = 0
    for k in range(400):
        m = int(rng.integers(1, 12))
        n = int(rng.integers(1, 60))
        odds = rng.integers(0, ODDS_SCALE + 1, (n, m)).astype(np.int16)
        if k % 3 == 0:
            odds[:, : m // 2] = odds[:, :1]                 # equal odds across entries
        price = []
        for j in range(m):
            kind = int(rng.integers(0, 4))
            if kind == 0:
                price.append(Fraction(int(rng.integers(0, 10**6)), 10**6))
            elif kind == 1:
                price.append(Fraction(int(rng.integers(1, 2**60)), 2**62 + int(rng.integers(1, 99))))
            elif kind == 2 and j:
                price.append(price[j - 1])                   # an equal price
            else:
                # a utility equal to entry 0's at some odds: p/S - c = p0/S - c0
                d = int(rng.integers(-50, 50))
                price.append((price[0] if price else Fraction(0)) + Fraction(d, ODDS_SCALE))
        have = [j for j in range(m) if rng.random() < 0.85]
        got = pick_rows(odds, have, price)
        for q in range(n):
            want = _exact_pick(odds[q], have, price)
            assert got[q] == want
            us = [Fraction(int(odds[q, j]), ODDS_SCALE) - price[j] for j in have]
            ties += len(us) != len(set(us))
    assert ties > 100
