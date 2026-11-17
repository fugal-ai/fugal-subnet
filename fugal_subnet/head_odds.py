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
from decimal import Decimal, localcontext

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
