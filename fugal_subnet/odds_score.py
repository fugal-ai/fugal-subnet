"""Scoring a head's odds on the exploration pool (protocol v3).

The seat goes to the head whose probabilities are honest and sharp: for a
(question, model) pair nobody chose, it said p, and the model was right or
wrong. The score is the Brier loss (p - y)^2, a proper scoring rule: a head
does best by saying what it believes, and better by believing what is true.

UNITS. p and y are integers in 0..ODDS_SCALE (head_odds.py; a pair's outcome
is a coldkey median, explore_pool.py), so a pair's loss is an integer in
0..ODDS_SCALE^2 and every sum is exact on any CPU. ODDS_SCALE^2 is one unit
of Brier score, so 0.01 of Brier score ("one Brier point") is exactly
champion.NV_UNITS_PER_POINT of these units, and the seat's paired test
(`champion.paired_test`) is used as it is, with a margin in Brier points.

THE REFERENCE R is what a head that does not read the question can say: each
model's success rate on each benchmark,

    p_R(m, b) = (k_mb + A * r_m) / (n_mb + A),    r_m = (k_m + 1) / (n_m + 2)

from the pooled pairs of the OTHER epochs in the window (leave one epoch out,
so R never sees the outcomes it is scored on), shrunk to the model's overall
rate with strength A (`ODDS_REF_PRIOR`), which is what keeps a benchmark seen
three times from saying 0% or 100%. A head's SKILL is how much lower its loss
is than R's on the same pairs. A table of rates by kind of question has about
zero skill against R by construction; skill is what reading the question adds.

A MODEL A HEAD HAS NO ROW FOR is scored at p_R: zero skill there, no penalty.

FRESH PAIRS ONLY. A head is scored on the pairs of epochs it was itself
scored in (its current unbroken run, like the net-value comparison). It was
committed on chain before each of those epochs' nonces existed, so it cannot
have been fitted to their outcomes; pairs from before its run are published
history that it may have trained on, and are never read for it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from fugal_subnet.champion import NV_UNITS_PER_POINT, EpochPairs, Window
from fugal_subnet.head_odds import ODDS_SCALE

# One Brier point (0.01 of Brier score) in loss units, and it must be the
# paired test's unit: the seat rule passes a margin in points to paired_test.
BRIER_UNITS_PER_POINT = ODDS_SCALE * ODDS_SCALE // 100
assert BRIER_UNITS_PER_POINT == NV_UNITS_PER_POINT


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


@dataclass(frozen=True)
class Reference:
    """R for every epoch of a window: `odds[e]` is p_R per pair of epoch e
    (int32, ODDS_SCALE units), from the pairs of every other window epoch."""

    odds: dict[int, np.ndarray]

    def of(self, epoch: int) -> np.ndarray:
        return self.odds[epoch]


def reference(pairs: Mapping[int, EpochPairs], prior: int) -> Reference:
    """Leave-one-epoch-out R over the window's pairs. Exact (Fractions)."""
    if prior < 0:
        raise ValueError(f"the reference prior must not be negative, got {prior!r}")
    k_mb: dict[tuple[str, str], int] = {}
    n_mb: dict[tuple[str, str], int] = {}
    k_m: dict[str, int] = {}
    n_m: dict[str, int] = {}
    per_epoch: dict[int, tuple[dict, dict, dict, dict]] = {}
    for e in sorted(pairs):
        ep = pairs[e]
        ek_mb: dict[tuple[str, str], int] = {}
        en_mb: dict[tuple[str, str], int] = {}
        ek_m: dict[str, int] = {}
        en_m: dict[str, int] = {}
        for model, bench, y in zip(ep.models, ep.benchmarks, ep.y.tolist()):
            cell = (model, bench)
            ek_mb[cell] = ek_mb.get(cell, 0) + int(y)
            en_mb[cell] = en_mb.get(cell, 0) + 1
            ek_m[model] = ek_m.get(model, 0) + int(y)
            en_m[model] = en_m.get(model, 0) + 1
        per_epoch[e] = (ek_mb, en_mb, ek_m, en_m)
        for total, part in ((k_mb, ek_mb), (n_mb, en_mb), (k_m, ek_m), (n_m, en_m)):
            for key, v in part.items():
                total[key] = total.get(key, 0) + v

    out: dict[int, np.ndarray] = {}
    for e in sorted(pairs):
        ep = pairs[e]
        ek_mb, en_mb, ek_m, en_m = per_epoch[e]
        cache: dict[tuple[str, str], int] = {}
        vals = np.empty(len(ep.models), dtype=np.int32)
        for i, (model, bench) in enumerate(zip(ep.models, ep.benchmarks)):
            cell = (model, bench)
            p = cache.get(cell)
            if p is None:
                # Counts with this epoch's own pairs taken out. k is a sum of
                # outcomes in ODDS_SCALE units.
                km = Fraction(k_m.get(model, 0) - ek_m.get(model, 0), ODDS_SCALE)
                nm = n_m.get(model, 0) - en_m.get(model, 0)
                kc = Fraction(k_mb.get(cell, 0) - ek_mb.get(cell, 0), ODDS_SCALE)
                nc = n_mb.get(cell, 0) - en_mb.get(cell, 0)
                r_m = (km + 1) / (nm + 2)
                rate = (kc + prior * r_m) / (nc + prior) if (nc + prior) else r_m
                p = cache[cell] = _round_half_up(rate * ODDS_SCALE)
            vals[i] = p
        out[e] = vals
    return Reference(out)


def expected_odds(pairs: Mapping[int, EpochPairs], prior: int,
                  cells: Mapping[tuple[str, str], str],
                  min_model_pairs: int = 0) -> dict[tuple[str, str], int]:
    """{(question_id, model): p_R} for pairs that are NOT in `pairs` yet.

    `cells` maps each new (question_id, model) to its benchmark; `pairs` is the
    window before the new epoch. This is what an operator's fresh exploration
    answers are judged against (explore_pool.private_implausible): history,
    never a pool the operator has just contributed to.

    A model with fewer than `min_model_pairs` pairs in the window has NO
    expectation and is absent from the result. With little or no history the
    reference is its prior, one half, and every honest operator on a 90%
    model would sit far "above" it: at a cold start the whole field would be
    flagged, and its exploration thrown away, for being right.
    """
    k_mb: dict[tuple[str, str], int] = {}
    n_mb: dict[tuple[str, str], int] = {}
    k_m: dict[str, int] = {}
    n_m: dict[str, int] = {}
    for ep in pairs.values():
        for model, bench, y in zip(ep.models, ep.benchmarks, ep.y.tolist()):
            k_mb[(model, bench)] = k_mb.get((model, bench), 0) + int(y)
            n_mb[(model, bench)] = n_mb.get((model, bench), 0) + 1
            k_m[model] = k_m.get(model, 0) + int(y)
            n_m[model] = n_m.get(model, 0) + 1
    cache: dict[tuple[str, str], int] = {}
    out: dict[tuple[str, str], int] = {}
    for (qid, model), bench in cells.items():
        if n_m.get(model, 0) < int(min_model_pairs):
            continue
        cell = (model, bench)
        if cell not in cache:
            r_m = (Fraction(k_m.get(model, 0), ODDS_SCALE) + 1) / (n_m.get(model, 0) + 2)
            nc = n_mb.get(cell, 0)
            rate = ((Fraction(k_mb.get(cell, 0), ODDS_SCALE) + prior * r_m) / (nc + prior)
                    if (nc + prior) else r_m)
            cache[cell] = _round_half_up(rate * ODDS_SCALE)
        out[(qid, model)] = cache[cell]
    return out


def loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pair Brier loss in integer units: (p - y)^2. Elementwise, int64."""
    d = p.astype(np.int64) - y.astype(np.int64)
    return d * d


def head_odds_on(window: Window, ref: Reference, head: str, epoch: int) -> np.ndarray:
    """`head`'s odds on `epoch`'s pairs (int64).

    A pair whose model the head has no row for (-1) is scored as certainly
    wrong, the worst loss there is: a head must give odds for every menu
    entry. Filling the gap with R's odds let a copy of the champion with its
    weakest rows removed beat the champion with no skill of its own, because
    R is fitted on the window's outcomes (v3 review, 2026-11-16). Such a head
    is also not eligible for the odds seat at all (`missing_rows`); `ref` is
    kept in the signature for the callers that compare against it."""
    del ref
    o = window.odds[head][epoch].astype(np.int64)
    y = window.pairs[epoch].y.astype(np.int64)
    certainly_wrong = np.where(2 * y >= ODDS_SCALE, 0, ODDS_SCALE)
    return np.where(o < 0, certainly_wrong, o)


def missing_rows(window: Window, head: str, since_epoch: int | None = None,
                 run_only: bool = True) -> int:
    """How many of the pairs `head` is scored on name a menu entry it gave no
    odds for. Any is disqualifying for the odds seat: from v3 a head gives odds
    for every menu entry (v2_epoch._pool_and_odds publishes a head that does
    not cover the menu with no odds at all)."""
    return int(sum(int((_kept(window.odds[head][e], window.pairs[e]) < 0).sum())
                   for e in scored_epochs(window, head, since_epoch, run_only)))


def _kept(values, ep: EpochPairs):
    """`values` (one per pair) without the pairs the odds score leaves out
    (EpochPairs.scored: v3 menu B's grace period)."""
    return values if ep.scored is None else values[ep.scored]


def scored_epochs(window: Window, head: str, since_epoch: int | None = None,
                  run_only: bool = True) -> list[int]:
    """The epochs `head`'s odds are scored on: epochs with pairs, in which it
    has odds, inside its current run (or, for a seated head, anywhere in the
    window: the same two readings as the net-value comparison)."""
    has = window.odds.get(head, {})
    epochs = window.current_run(head, since_epoch) if run_only else window.epochs_of(head, since_epoch)
    return [e for e in epochs if e in window.pairs and e in has]


@dataclass(frozen=True)
class OddsUnits:
    by_category: dict[str, np.ndarray]
    n_pairs: int

    @property
    def total(self) -> int:
        return int(sum(int(x.sum()) for x in self.by_category.values()))


def _by_category(units: list[np.ndarray], cats: list[np.ndarray]) -> OddsUnits:
    if not units:
        return OddsUnits({}, 0)
    u, c = np.concatenate(units), np.concatenate(cats)
    return OddsUnits({name: u[c == name] for name in sorted(set(c.tolist()))}, int(u.size))


def _weighted(units: np.ndarray, ep: EpochPairs) -> np.ndarray:
    """Per-pair units times the pair's exploration weight (explore_weights):
    pairs of entries drawn less often count for more, so the sum estimates what
    uniform sampling would have. Exact integers; a pair count stays a count.
    Pairs the odds score leaves out (EpochPairs.scored) are dropped."""
    if ep.weight is not None:
        from fugal_subnet.explore_weights import apply
        units = apply(units, ep.weight)
    return _kept(units, ep)


def skill_units(window: Window, ref: Reference, head: str, since_epoch: int | None = None,
                run_only: bool = True) -> OddsUnits:
    """Per pair, R's loss minus `head`'s: positive where the head beat R."""
    units, cats = [], []
    for e in scored_epochs(window, head, since_epoch, run_only):
        ep = window.pairs[e]
        units.append(_weighted(loss(ref.of(e), ep.y) - loss(head_odds_on(window, ref, head, e), ep.y), ep))
        cats.append(_kept(np.asarray(ep.categories), ep))
    return _by_category(units, cats)


def brier_units(window: Window, ref: Reference, challenger: str, bar: str,
                since_epoch: int | None = None) -> OddsUnits:
    """Per pair, the bar's loss minus the challenger's, on the pairs of the
    challenger's current run in which the bar was also scored: positive where
    the challenger's odds were closer to what happened."""
    bar_epochs = set(scored_epochs(window, bar, None, run_only=False))
    units, cats = [], []
    for e in scored_epochs(window, challenger, since_epoch):
        if e not in bar_epochs:
            continue
        ep = window.pairs[e]
        units.append(_weighted(loss(head_odds_on(window, ref, bar, e), ep.y)
                               - loss(head_odds_on(window, ref, challenger, e), ep.y), ep))
        cats.append(_kept(np.asarray(ep.categories), ep))
    return _by_category(units, cats)


def solo_skill(window: Window, ref: Reference, head: str,
               since_epoch: int | None = None) -> tuple[Fraction, int]:
    """(`head`'s mean skill in loss units as an exact Fraction, pairs) over its run."""
    u = skill_units(window, ref, head, since_epoch)
    return (Fraction(u.total, u.n_pairs) if u.n_pairs else Fraction(0)), u.n_pairs


def points(units: Fraction | int | float) -> float:
    """Loss units as Brier points, for logs and the reveal."""
    return float(units) / BRIER_UNITS_PER_POINT
