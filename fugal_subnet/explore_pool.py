"""The exploration pool: what the field measured this epoch (protocol v3).

Every miner answers exploration pairs it did not choose: a question sent to a
model the epoch nonce picked (the shared set, the same for everyone) or the
nonce and its hotkey picked (its private set). Pooled, they are the subnet's
only unbiased sample of how each model does on each kind of question, and in
v3 they are what a head's odds are scored on (fugal_subnet/odds_score.py).

One pooled PAIR per (question, model) answered this epoch, built the way a
shared grade is (shared_grades.py), so one operator is one vote however many
hotkeys it runs:

  1. failed calls are dropped: a call that did not complete says nothing
     about the model;
  2. operators flagged this epoch are dropped (`private_implausible`, and the
     validator's existing exploration detectors);
  3. each operator's PRIVATE answers are capped at `coldkey_cap`, keeping the
     ones with the lowest keyed digest, so an operator with fifty hotkeys does
     not supply fifty times the pool one with a single hotkey does;
  4. one vote per coldkey per pair (the mean over its hotkeys), and the pair's
     outcome y is the MEDIAN across coldkeys, in units of 1 / ODDS_SCALE.

WHAT AN OPERATOR CAN DO TO IT (I4), stated plainly. A private pair usually has
one voter, so its outcome is that operator's attested answer: a model call
made inside its TD and graded by the pinned grader. An operator that controls
its TD's network can make calls fail (dropped here, and bounded by
EXPLORE_FAIL_MARGIN in the proof check), but cannot choose the pair, the
answer or the grade. What it contributes is bounded by the cap, and an
operator whose private answers are implausible for the models it was sent to
has them left out. A flag never touches that miner's own score.

Everything is exact (integers and Fractions) and order-independent (I1).
"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from fugal_subnet.head_odds import ODDS_SCALE


@dataclass(frozen=True)
class ExploreAnswer:
    """One exploration result from one verified proof."""

    question_id: str
    model: str
    coldkey: str
    correct: bool
    failed: bool
    private: bool                 # from the miner's private set (else the shared set)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0           # attested wall time of the call (proof.QuestionResult)


@dataclass(frozen=True)
class Pair:
    """One pooled (question, model) observation of one epoch."""

    question_id: str
    model: str
    y: int                        # 0..ODDS_SCALE: the coldkey-median outcome
    voters: int                   # coldkeys that answered it
    prompt_tokens: int            # coldkey-median attested tokens, rounded
    completion_tokens: int
    # The pair's inverse-probability weight in explore_weights.WEIGHT_SCALE
    # units: exploration draws entries by cost, and the odds score weighs each
    # pair back to what uniform sampling would have counted.
    weight: int = 1_000_000
    latency_ms: int = 0           # coldkey-median attested call time, for routers to learn from


def _round_half_up(x: Fraction) -> int:
    return int((2 * x + 1) // 2)


def _median(values: list[Fraction]) -> Fraction:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _cap_digest(nonce: bytes, a: ExploreAnswer) -> bytes:
    return hmac.new(nonce, f"cap:{a.coldkey}:{a.question_id}:{a.model}".encode("utf-8"),
                    hashlib.sha256).digest()


def capped(answers: Iterable[ExploreAnswer], nonce: bytes, coldkey_cap: int) -> list[ExploreAnswer]:
    """`answers` with each coldkey's PRIVATE pairs cut to `coldkey_cap`.

    The pairs kept are the ones with the lowest HMAC(nonce, coldkey, question,
    model): a fixed function of public values, so every validator keeps the
    same ones and the operator cannot choose which. All of an operator's
    answers to a kept pair stay (they are one vote). Shared answers are never
    cut: every operator answers the same ones.
    """
    answers = list(answers)
    by_ck: dict[str, set[tuple[str, str]]] = {}
    digest: dict[tuple[str, str, str], bytes] = {}
    for a in answers:
        if a.private:
            cell = (a.question_id, a.model)
            by_ck.setdefault(a.coldkey, set()).add(cell)
            digest[(a.coldkey, *cell)] = _cap_digest(nonce, a)
    keep: dict[str, set[tuple[str, str]]] = {}
    for ck, cells in by_ck.items():
        ranked = sorted(cells, key=lambda cell: (digest[(ck, *cell)], cell))
        keep[ck] = set(ranked[:max(0, int(coldkey_cap))])
    return [a for a in answers
            if not a.private or (a.question_id, a.model) in keep[a.coldkey]]


def pool_pairs(answers: Iterable[ExploreAnswer], *, nonce: bytes, coldkey_cap: int,
               flagged: Iterable[str] = (),
               weights: Mapping[str, int] | None = None) -> list[Pair]:
    """The epoch's pooled pairs, sorted by (question, model).

    `weights` is {entry: inverse-probability weight} of the epoch's exploration
    shares (explore_weights.weights); None weighs every pair one."""
    from fugal_subnet.explore_weights import WEIGHT_SCALE
    out_of = set(flagged)
    live = [a for a in answers if not a.failed and a.coldkey not in out_of]
    live = capped(live, nonce, coldkey_cap)
    by_cell: dict[tuple[str, str], dict[str, list[ExploreAnswer]]] = {}
    for a in live:
        by_cell.setdefault((a.question_id, a.model), {}).setdefault(a.coldkey, []).append(a)
    pairs = []
    for (qid, model), votes in sorted(by_cell.items()):
        grade = [Fraction(sum(bool(a.correct) for a in v), len(v)) for v in votes.values()]
        p_tok = [Fraction(sum(int(a.prompt_tokens) for a in v), len(v)) for v in votes.values()]
        c_tok = [Fraction(sum(int(a.completion_tokens) for a in v), len(v)) for v in votes.values()]
        lat = [Fraction(sum(int(a.latency_ms) for a in v), len(v)) for v in votes.values()]
        pairs.append(Pair(
            question_id=qid, model=model,
            y=_round_half_up(_median(grade) * ODDS_SCALE), voters=len(votes),
            prompt_tokens=_round_half_up(_median(p_tok)),
            completion_tokens=_round_half_up(_median(c_tok)),
            weight=int(weights[model]) if weights is not None else WEIGHT_SCALE,
            latency_ms=_round_half_up(_median(lat)),
        ))
    return pairs


# The variance of one answer is floored as if its expected accuracy were no
# nearer 0 or 1 than this (in ODDS_SCALE units). A reference that says 99.7%
# would otherwise call two honest misses in 24 answers implausible.
_VARIANCE_CLAMP = 500


def private_implausible(
    answers: Iterable[ExploreAnswer],
    expected: Mapping[tuple[str, str], int],
    *,
    z: int,
    min_samples: int,
) -> dict[str, str]:
    """{coldkey: why} for operators whose private answers are implausible.

    `expected[(question_id, model)]` is the reference odds for that pair, in
    ODDS_SCALE units (odds_score.Reference, built from EARLIER epochs so an
    operator is judged against history, not against a pool it has just
    contributed to). Over an operator's non-failed private answers, with
    outcomes y_i and expectations p_i, it is flagged when

        (sum y_i - sum p_i)^2  >  z^2 * sum p_i (1 - p_i)

    in exact integers: its accuracy is more than z standard deviations from
    what those models do on those kinds of question, in either direction.
    Fewer than `min_samples` judged answers is never a flag. The consequence
    is only that its pairs stay out of the pool this epoch (the caller passes
    the keys to `pool_pairs`); its own score is untouched (I4).
    """
    stats: dict[str, list[int]] = {}
    for a in answers:
        if a.failed or not a.private:
            continue
        p = expected.get((a.question_id, a.model))
        if p is None:
            continue
        s = stats.setdefault(a.coldkey, [0, 0, 0])
        v = min(max(int(p), _VARIANCE_CLAMP), ODDS_SCALE - _VARIANCE_CLAMP)
        s[0] += (ODDS_SCALE if a.correct else 0) - int(p)
        s[1] += v * (ODDS_SCALE - v)
        s[2] += 1
    out: dict[str, str] = {}
    for ck in sorted(stats):
        diff, var, n = stats[ck]
        if n >= int(min_samples) and diff * diff > int(z) * int(z) * var:
            direction = "above" if diff > 0 else "below"
            out[ck] = (f"private exploration accuracy is {abs(diff) / ODDS_SCALE:.1f} answers "
                       f"{direction} the reference over {n} answers (more than {z} standard "
                       "deviations)")
    return out


def pairs_for_reveal(pairs: Iterable[Pair], benchmark_of: Mapping[str, str]) -> list[dict]:
    """The pool as the reveal publishes it: sorted, with each question's
    benchmark, so a rebuild needs no pool file to place a pair."""
    return [
        {"question_id": p.question_id, "model": p.model,
         "benchmark": str(benchmark_of.get(p.question_id, "")), "y": int(p.y),
         "voters": int(p.voters), "prompt_tokens": int(p.prompt_tokens),
         "completion_tokens": int(p.completion_tokens), "weight": int(p.weight),
         "latency_ms": int(p.latency_ms)}
        for p in sorted(pairs, key=lambda p: (p.question_id, p.model))
    ]
