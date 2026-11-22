"""A head's odds, computed by the validator, identical on every host (I1).

Protocol v3 scores what a head PREDICTS: for a question and a model, the
chance the model answers it correctly. The prediction is a function of two
pinned things, the head's bytes (attested by hash) and the question's
embedding (the pinned embeddings file), so a validator computes it itself.
Nothing about it is the miner's report, and the proof needs no new field.

The value is an integer p in 0..ODDS_SCALE, meaning p / ODDS_SCALE: the
logistic of z = w . h + b, rounded to the nearest 1e-4. It has to be the same
integer on every validator, because per-pair losses are summed exactly and
compared against a margin. A float64 sigmoid is not that: `exp` comes from the
platform's math library, and a dot product's value depends on the order and
width of the additions (BLAS kernel, SIMD, FMA). So:

- **The products are exact.** Each w_i * h_i is one IEEE-754 double multiply
  (elementwise, never a reduction), which is correctly rounded everywhere; for
  float32 weights and embeddings the product of two 24-bit significands fits
  a double exactly, so nothing is rounded at all.
- **The sum is `math.fsum`**, the correctly rounded sum of its arguments,
  independent of their order.
- **No exponential is evaluated.** z is compared with a table of thresholds,
  T[k] = logit((k + 0.5) / ODDS_SCALE) for k = 0..ODDS_SCALE-1, and p is the
  number of thresholds at or below z. The table is built once with `decimal`
  arithmetic (correctly rounded by specification, so no platform library is
  involved) and its digest is pinned: a build that produced a different table
  refuses to import.

`HEAD_MAX_ABS_PARAM` bounds |w| and |b|, checked when a head is validated, so
z is finite and far inside a double's range (|h| is at most 1 per row).
"""
from __future__ import annotations

import bisect
import hashlib
import math
import struct
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

ODDS_SCALE = 10_000
# A weight or bias larger than this in magnitude is refused when the head is
# loaded (protocol v3). A trained logistic head is orders of magnitude inside
# it; the bound exists so that every sum here is finite and exact.
HEAD_MAX_ABS_PARAM = 1_000_000.0
# sha256 of the threshold table's doubles, big-endian, in order.
THRESHOLDS_SHA256 = "d9c75d9418d2d6305fc17c33b26c7efd98d9158a91dc27d3f000427fa41ad8b0"


def _build_thresholds() -> tuple[float, ...]:
    out = []
    with localcontext() as ctx:
        # Far more digits than a double holds; the table is the same at 40,
        # 60 and 100 (tests/test_head_odds.py), so this is not a tuning knob.
        ctx.prec = 40
        scale = Decimal(ODDS_SCALE)
        for k in range(ODDS_SCALE):
            p = (Decimal(k) + Decimal("0.5")) / scale
            # float(Decimal) is a correctly rounded conversion.
            out.append(float((p / (1 - p)).ln()))
    return tuple(out)


def thresholds_digest(table) -> str:
    return hashlib.sha256(b"".join(struct.pack(">d", t) for t in table)).hexdigest()


_THRESHOLDS: tuple[float, ...] | None = None


def thresholds() -> tuple[float, ...]:
    """The pinned table, built on first use (about half a second) and checked
    against its digest: a host that builds a different table raises here
    rather than score odds differently from every other validator."""
    global _THRESHOLDS
    if _THRESHOLDS is None:
        table = _build_thresholds()
        got = thresholds_digest(table)
        if got != THRESHOLDS_SHA256:
            raise RuntimeError(
                "head_odds: the logit threshold table differs from the pinned one "
                f"({got} != {THRESHOLDS_SHA256}); this host would score odds differently "
                "from every other validator"
            )
        _THRESHOLDS = table
    return _THRESHOLDS


def logit_exact(w_row: np.ndarray, bias, hidden: np.ndarray) -> float:
    """z = w . h + b, the correctly rounded sum of exact products."""
    w = np.asarray(w_row, dtype=np.float64)
    h = np.asarray(hidden, dtype=np.float64)
    if w.ndim != 1 or w.shape != h.shape:
        raise ValueError(f"weight row {w.shape} and embedding {h.shape} do not align")
    # Elementwise only: each product is one IEEE multiply, identical on any CPU.
    return math.fsum([*(w * h).tolist(), float(bias)])


def odds_of_logit(z: float) -> int:
    """The integer odds for logit z: how many thresholds are at or below it."""
    if z != z:
        raise ValueError("logit is not a number")
    return bisect.bisect_right(thresholds(), z)


def odds(w_row: np.ndarray, bias, hidden: np.ndarray) -> int:
    """P(the model answers correctly), in units of 1 / ODDS_SCALE."""
    return odds_of_logit(logit_exact(w_row, bias, hidden))


# --- many odds at once ----------------------------------------------------------------------
#
# `odds` evaluates one exact 1,024-term sum in Python per (question, row);
# a validator needs millions an epoch. `odds_matrix` gives the same integers
# by construction: it computes every logit with float64 matrix products, and
# keeps a value only where it is provably on the same side of every
# threshold as the exact logit; the rest go through `odds`.
#
# THE ERROR BOUND. Let u = 2^-53, n the width (1,024), for one (row,
# question) M = sum_j |w_j h_j| and P = sum_j w_j h_j + b the true value.
#   exact path:  z_ex = fsum(fl(w_j h_j), b). Each product is one IEEE
#                multiply, |fl(w_j h_j) - w_j h_j| <= u |w_j h_j| (0 for
#                float32 inputs), and fsum rounds the exact sum once, so
#                |z_ex - P| <= u M + u (|P| + u M) <= 2u (M + |b|) + u^2 M.
#   fast path:   z = fl(dot + b), dot a float64 sum of the n products in some
#                order, with or without fused multiply-adds (a BLAS kernel):
#                |dot - (P - b)| <= gamma_n M, gamma_n = n u / (1 - n u)
#                (Higham, Accuracy and Stability, 3.1, any order), and the
#                bias adds u |dot + b| <= u (M (1 + gamma_n) + |b|).
#   so |z - z_ex| <= (gamma_n + 4u) M + 2u |b| + ...  <= 2u (n + 8) (M + |b|)
#   for n >= 5 (gamma_n <= 1.01 n u), with room to spare.
# M is bounded above without squaring (no underflow): M <= ||w||_1 ||h||_inf,
# ||w||_1 summed in float64 (relative error <= gamma_n) and inflated by
# 2^-30, ||h||_inf exact. Every inequality above assumes no underflow; each
# of the 2n + 2 roundings can add at most 2^-1022 absolutely if a kernel
# flushes subnormals, covered by (n + 8) 2^-1000. The band computed is twice
# all that plus 2^-50 |z| (the rounding of z +- E itself), so the real
# interval [z - E, z + E] certainly holds z_ex.
#
# THE DECISION. The odds of a logit are the number of thresholds at or below
# it, monotone in the logit, so with lo = that count at z - E and hi = it at
# z + E, the exact odds lie in [lo, hi]; where lo == hi they ARE lo. Where
# they differ (a threshold within E of z: for a trained head, |W| <= 1e6,
# |b| <= 1e6 and unit embeddings, E is at most about 1e-4 and usually about
# 1e-12, against threshold gaps of at least 4e-4), or where anything is not
# finite, the exact `odds` is evaluated instead. So every integer returned is
# `odds`' integer, whatever the BLAS, its threads or its summation order.

_U = 2.0 ** -53


def _band(z: np.ndarray, w_l1: np.ndarray, h_inf: np.ndarray, bias: np.ndarray, n: int):
    """The half-width E (module comment) for logits z (questions x rows)
    with the rows' L1 norms `w_l1`, the questions' max-norms `h_inf` and
    the rows' biases."""
    m_up = (h_inf[:, None] * (w_l1 * (1.0 + 2.0 ** -30))[None, :])
    with np.errstate(over="ignore", invalid="ignore"):
        e = 2.0 * (2.0 * _U * (n + 8) * (m_up + np.abs(bias)[None, :])
                   + (n + 8) * 2.0 ** -1000) + 2.0 ** -50 * np.abs(z)
    return e


def _settle(z: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(odds, settled): the odds where every threshold is outside
    [z - E, z + E], and whether it is."""
    t = np.asarray(thresholds(), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        lo = np.searchsorted(t, z - e, side="right")
        hi = np.searchsorted(t, z + e, side="right")
        ok = (lo == hi) & np.isfinite(z) & np.isfinite(e)
    return lo, ok


def odds_matrix(W: np.ndarray, b, rows, hidden) -> np.ndarray:
    """(questions, len(rows)) int16: `odds(W[r], b[r], hidden[q])` for every
    question q and every row index r in `rows`, identical by construction
    (the module comment above): float64 products where the exact value is
    provably on the same side of every threshold, `odds` elsewhere."""
    rows = [int(r) for r in rows]
    H = np.asarray(hidden)
    W = np.asarray(W)
    out = np.empty((H.shape[0] if H.ndim == 2 else len(hidden), len(rows)), dtype=np.int16)
    if not rows or out.shape[0] == 0:
        return out
    if H.ndim != 2 or W.ndim != 2 or H.shape[1] != W.shape[1]:
        # not one embedding per question of the rows' width: odds() says why
        for q in range(out.shape[0]):
            for j, r in enumerate(rows):
                out[q, j] = odds(W[r], b[r], hidden[q])
        return out
    H64 = np.asarray(H, dtype=np.float64)
    Wr = np.asarray(W[rows], dtype=np.float64)
    br = np.array([float(b[r]) for r in rows], dtype=np.float64)
    n = H64.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        z = H64 @ Wr.T + br[None, :]
        e = _band(z, np.abs(Wr).sum(axis=1), np.abs(H64).max(axis=1), br, n)
    p, ok = _settle(z, e)
    out[:] = p
    for q, j in zip(*np.nonzero(~ok)):
        out[q, j] = odds(W[rows[j]], b[rows[j]], hidden[q])
    return out


def _stacked(hidden_rows, n: int):
    """The pairs' embeddings as one 2-D array, or None when they are not
    one row each of one width (then `odds` itself is evaluated, and raises
    what it raises)."""
    try:
        H = hidden_rows if isinstance(hidden_rows, np.ndarray) else np.asarray(hidden_rows)
    except ValueError:
        return None
    if H.ndim != 2 or len(H) != n or H.dtype.kind not in "fiu":
        return None
    return H


def odds_grid(heads, codes, hidden_rows) -> np.ndarray:
    """(heads, pairs) int16: for head k = (W, b, row_of_code) and pair i,
    `odds(W[r], b[r], hidden_rows[i])` with r = row_of_code[codes[i]], -1
    where that is negative or `codes[i]` is (no row). Pairs sharing a code
    read the same row of every head, so each code's pairs are one matrix
    product with every head's row for it (`odds_matrix`'s computation)."""
    codes = np.asarray(codes, dtype=np.int64)
    heads = list(heads)
    out = np.full((len(heads), len(codes)), -1, dtype=np.int16)
    if not len(codes) or not heads:
        return out
    H = _stacked(hidden_rows, len(codes))
    if H is None:
        for k, (W, b, row_of_code) in enumerate(heads):
            for i, c in enumerate(codes.tolist()):
                r = -1 if c < 0 else int(row_of_code[c])
                if r >= 0:
                    out[k, i] = odds(W[r], b[r], hidden_rows[i])
        return out
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
    for ix in np.split(order, cuts):
        c = int(codes[ix[0]])
        if c < 0:
            continue
        mine = [(k, int(row_of_code[c])) for k, (_W, _b, row_of_code) in enumerate(heads)
                if int(row_of_code[c]) >= 0]
        if not mine:
            continue
        Hc = H[ix]
        if all(np.asarray(heads[k][0]).ndim == 2
               and np.asarray(heads[k][0]).shape[1] == H.shape[1] for k, _r in mine):
            Wc = np.stack([np.asarray(heads[k][0][r], dtype=np.float64) for k, r in mine])
            bc = np.array([float(heads[k][1][r]) for k, r in mine], dtype=np.float64)
            H64 = np.asarray(Hc, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                z = H64 @ Wc.T + bc[None, :]
                e = _band(z, np.abs(Wc).sum(axis=1), np.abs(H64).max(axis=1), bc, H64.shape[1])
            p, ok = _settle(z, e)
            for j, (k, r) in enumerate(mine):
                out[k, ix] = p[:, j]
                for q in np.flatnonzero(~ok[:, j]).tolist():
                    out[k, ix[q]] = odds(heads[k][0][r], heads[k][1][r], hidden_rows[ix[q]])
        else:
            for k, r in mine:
                for q in ix.tolist():
                    out[k, q] = odds(heads[k][0][r], heads[k][1][r], hidden_rows[q])
    return out


def odds_by_row(W: np.ndarray, b, row_of_pair, hidden_rows) -> np.ndarray:
    """(pairs,) int16: `odds(W[r], b[r], hidden_rows[i])` with r =
    row_of_pair[i], -1 where it is negative (no row): `odds_grid` with one
    head whose code is its row."""
    row_of_pair = np.asarray(row_of_pair, dtype=np.int64)
    n_rows = int(row_of_pair.max()) + 1 if len(row_of_pair) else 0
    return odds_grid([(W, b, np.arange(n_rows))], row_of_pair, hidden_rows)[0]


def check_head_bounds(W: np.ndarray, b: np.ndarray, limit: float = HEAD_MAX_ABS_PARAM) -> None:
    """Refuse a head with a weight or bias outside +/- `limit` (or not finite)."""
    for name, a in (("W", W), ("b", b)):
        a = np.asarray(a)
        if not np.isfinite(a).all():
            raise ValueError(f"head {name} holds a value that is not finite")
        if a.size and float(np.abs(a).max()) > float(limit):
            raise ValueError(f"head {name} holds a value larger than {limit:g} in magnitude")


def check_embedding_bounds(hidden: np.ndarray, limit: float = 1.0 + 1e-3) -> None:
    """Refuse an embeddings table with a value that is not finite or is larger
    than a unit vector's entries can be. With the head bound this keeps every
    logit finite: |z| <= limit * sum|w| + |b|."""
    h = np.asarray(hidden)
    if not np.isfinite(h).all():
        raise ValueError("embeddings hold a value that is not finite")
    if h.size and float(np.abs(h).max()) > float(limit):
        raise ValueError(f"embeddings hold a value larger than {limit:g} in magnitude")


@dataclass(frozen=True)
class MenuOdds:
    """A v4 head's odds and picks on one epoch's menu (`picks_and_odds_v4`)."""

    entries: tuple[str, ...]          # the menu's entry ids, in the state's order
    odds: np.ndarray                  # (questions, entries) int16; -1 where the head has no row
    picks: tuple[str | None, ...]     # per question; None when no entry on the menu has a row


def menu_rows(head_models, entry_ids) -> tuple[int | None, ...]:
    """For each entry id in `entry_ids`, the index of the head's row keyed by
    it, or None. Rows are matched by id only: their order in the head is
    irrelevant, and a row whose id is not in `entry_ids` is never read."""
    row_of: dict[str, int] = {}
    for i, m in enumerate(head_models):
        if str(m) in row_of:
            raise ValueError(f"the head has two rows for {str(m)!r}")
        row_of[str(m)] = i
    return tuple(row_of.get(str(e)) for e in entry_ids)


def _lambda(lam) -> Fraction:
    from fugal_subnet.routing_protocol import BENCHMARK_LAMBDA

    try:
        out = Fraction(BENCHMARK_LAMBDA if lam is None else lam)
    except (ValueError, OverflowError, TypeError) as e:
        raise ValueError(f"lambda must be a finite non-negative number: {lam!r}") from e
    if out < 0:
        raise ValueError(f"lambda must be a finite non-negative number: {lam!r}")
    return out


def _v4_inputs(head, state):
    """(entry ids in the state's order, the head's row per entry, the exact
    pick costs: menu_state.pick_costs, what a call costs on the questions
    routers send the entry; never the call cost exploration reads)."""
    from fugal_subnet import menu_state as ms

    if not isinstance(state, ms.MenuState):
        raise ValueError("v4 odds are computed on an epoch's menu (a menu_state.MenuState)")
    ids = state.ids()
    return ids, menu_rows(head.models, ids), ms.pick_costs(state)


def picks_and_odds_v4(head, state, hidden: np.ndarray, *, lam=None,
                      exclude: frozenset = frozenset()) -> MenuOdds:
    """A v4 head's odds and picks on the epoch's menu: the one function the
    TD's harness routes with and a validator recomputes with
    (docs/decisions/LIVE_MENU_V4.md, 7.3 and 11).

    `head` is a head_eval.HeadArtifact (from `load_head_v4`), `state` the
    epoch's menu_state.MenuState and `hidden` the questions' embeddings,
    (questions, 1024).

    - **Odds** are computed for the state's entries, in the state's order,
      each with the head's row keyed by that entry's id: `odds`, the exact
      integer of v3. An entry with no row has no odds (-1); in scoring it
      is the entry's measured rate during its allowance and certainly
      wrong after (labels_v4). A row for an id not on the menu is never
      read. A request update keeps the entry's id, and so its row.
    - **The pick** maximises odds / ODDS_SCALE - lam x cost over the entries
      with odds, where cost is the state's pick cost of the entry
      (menu_state.pick_costs: prices x the token means of its graded
      scored-question cells, shrunk toward its call cost; the call cost
      itself for an entry with no graded cells) and lam the benchmark
      setting (routing_protocol.BENCHMARK_LAMBDA unless given). The call
      cost (menu_state.costs) prices exploration, never a pick.
      Every term is a Fraction, so the comparison is exact; equal utilities
      go to the first entry in the state's order. Token means or a cost
      profile carried by the head are not read. A question with no entry
      with odds has no pick (None).
    - `exclude`: entry ids whose rows are not read (no odds, never picked):
      a challenge's picks leave out the provisional entries
      (picks_v4.challenge_picks).
    """
    ids, rows, costs = _v4_inputs(head, state)
    if exclude:
        rows = [None if m in exclude else r for m, r in zip(ids, rows)]
    rate = _lambda(lam)
    H = np.asarray(hidden)
    W, b = np.asarray(head.W), np.asarray(head.b)
    if H.ndim != 2 or W.ndim != 2 or H.shape[1] != W.shape[1]:
        raise ValueError(f"embeddings {H.shape} do not align with the head's rows {W.shape}")
    out = np.full((H.shape[0], len(ids)), -1, dtype=np.int16)
    have = [j for j, r in enumerate(rows) if r is not None]
    if have:
        out[:, have] = odds_matrix(W, b, [rows[j] for j in have], H)
    price = [rate * costs[e] for e in ids]
    best = pick_rows(out, have, price)
    picks = tuple(None if j < 0 else ids[j] for j in best.tolist())
    return MenuOdds(entries=ids, odds=out, picks=picks)


def _exact_pick(odds_row, cands, price) -> int:
    """argmax over `cands` (in order) of odds / ODDS_SCALE - price, each a
    Fraction; strict, so a tie keeps the earlier entry. -1 for none."""
    best, best_u = -1, None
    for j in cands:
        u = Fraction(int(odds_row[j]), ODDS_SCALE) - price[j]
        if best_u is None or u > best_u:
            best, best_u = j, u
    return best


def pick_rows(odds: np.ndarray, have, price) -> np.ndarray:
    """Per question (row of `odds`, int16 over the menu's entries), the index
    of the entry maximising odds / ODDS_SCALE - price[j] over the entries in
    `have` (those with a row), exactly as the Fraction comparison of
    `picks_and_odds_v4` decides it, ties to the earlier entry; -1 where
    `have` is empty.

    Computed in float64 first. Each utility in float64 is within delta of
    the exact one, delta <= 2^-51 (1 + max |price|) (a quotient by
    ODDS_SCALE, a price rounded to float64 and one subtraction, each one
    rounding); so the exact maximum, and every entry tied with it, is
    within 2 delta of the float maximum. Where only one entry is within TOL
    (far above 2 delta) of the float maximum it is the exact pick; where
    more are, the exact Fractions decide among those, in order."""
    have = [int(j) for j in have]
    n = odds.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    if not have or n == 0:
        return out
    pf = np.array([float(price[j]) for j in have], dtype=np.float64)
    tol = 2.0 ** -30 * (1.0 + float(np.abs(pf).max()))
    u = odds[:, have].astype(np.float64) / ODDS_SCALE - pf[None, :]
    top = u.max(axis=1)
    near = u >= (top - tol)[:, None]
    count = near.sum(axis=1)
    first = np.argmax(near, axis=1)
    hv = np.array(have, dtype=np.int64)
    out[:] = hv[first]
    tied = np.flatnonzero(count > 1)
    if len(tied):
        # Entries of one price are decided by their odds alone (the larger;
        # equal odds, the earlier entry): per question only the best entry of
        # each price among the near ones goes to the Fractions, in order.
        cls_of: dict = {}
        cls = np.array([cls_of.setdefault(price[j], len(cls_of)) for j in have], dtype=np.int64)
        for q in tied.tolist():
            k = np.flatnonzero(near[q])
            if len(np.unique(cls[k])) == 1:
                p = odds[q, hv[k]]
                out[q] = hv[k[int(np.argmax(p))]]     # first of the largest
                continue
            best: dict[int, int] = {}
            row = odds[q]
            for i in k.tolist():
                c, j = int(cls[i]), int(hv[i])
                if c not in best or int(row[j]) > int(row[best[c]]):
                    best[c] = j
            out[q] = _exact_pick(row, sorted(best.values()), price)
    return out


def pair_odds_v4(head, state, hidden_rows, pair_entries) -> np.ndarray:
    """A v4 head's odds on (question, entry) pairs, as int16: the same cells
    as `picks_and_odds_v4` (the same rows by id, the same exact odds), for a
    validator scoring pooled pairs without computing every entry of every
    question. Every pair's entry must be on the epoch's menu; one without a
    row gets -1."""
    ids, rows, _costs = _v4_inputs(head, state)
    if len(pair_entries) != len(hidden_rows):
        raise ValueError("one embedding row per pair is required")
    row_of = dict(zip(ids, rows))
    row_of_pair = []
    for eid in pair_entries:
        if str(eid) not in row_of:
            raise ValueError(f"{eid!r} is not on the menu of epoch {state.epoch}")
        r = row_of[str(eid)]
        row_of_pair.append(-1 if r is None else r)
    return odds_by_row(head.W, head.b, row_of_pair, hidden_rows)


def pair_odds_v4_many(heads, state, hidden_rows, pair_entries) -> list[np.ndarray]:
    """[pair_odds_v4(head, state, hidden_rows, pair_entries) for head in
    heads], computed together (`odds_grid`): the same integers."""
    heads = list(heads)
    if not heads:
        return []
    ids, _rows, _costs = _v4_inputs(heads[0], state)       # what pair_odds_v4 checks first
    if len(pair_entries) != len(hidden_rows):
        raise ValueError("one embedding row per pair is required")
    code_of = {e: j for j, e in enumerate(ids)}
    codes = []
    for eid in pair_entries:
        if str(eid) not in code_of:
            raise ValueError(f"{eid!r} is not on the menu of epoch {state.epoch}")
        codes.append(code_of[str(eid)])
    grid = []
    for head in heads:
        rows = menu_rows(head.models, ids)
        grid.append((head.W, head.b, np.array([-1 if r is None else r for r in rows],
                                              dtype=np.int64)))
    out = odds_grid(grid, codes, hidden_rows)
    return [out[k] for k in range(len(heads))]


def head_odds(W: np.ndarray, b: np.ndarray, head_models: list[str],
              hidden_rows: np.ndarray, pair_models: list[str]) -> np.ndarray:
    """One head's odds on a list of (question, model) pairs, as int16.

    `hidden_rows[i]` is the embedding of pair i's question and `pair_models[i]`
    its model. A model the head has no row for gets -1: the head makes no
    prediction there, and the scoring rule decides what that is worth.
    """
    if len(pair_models) != len(hidden_rows):
        raise ValueError("one embedding row per pair is required")
    row_of = {m: i for i, m in enumerate(head_models)}
    out = np.full(len(pair_models), -1, dtype=np.int16)
    for i, model in enumerate(pair_models):
        r = row_of.get(model)
        if r is not None:
            out[i] = odds(W[r], b[r], hidden_rows[i])
    return out
