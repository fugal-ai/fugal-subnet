"""Scoring: headroom above the constant-policy frontier.

    headroom = wilson_lcb(accuracy) - frontier(cost per question)
    score    = max(0, headroom) * burn_in * frontier_confidence
               (0 if wilson_lcb < SCORE_QUALITY_FLOOR * frontier max accuracy)

A router is paid for the value of reading the question: the accuracy it adds
over the best non-routing policy available at its own price. That reference is
the upper convex hull of what every single model — and every random mixture
of models — achieves at each cost, built from the reference frame's
nonce-assigned exploration samples and the pinned price table
(`fugal_subnet/frontier.py`). Every constant policy therefore scores zero by
construction; a router scores its headroom; cheap-and-smart routing, where the
frontier is low and steep, is where headroom is largest. That is the product,
stated as arithmetic.

**Why not the best single model.** The previous score compared a miner's
quality per dollar against ONE point of the frontier — the most accurate
model. Measured (docs/HEAD_EFFICACY.md): a head with W=0 that always called
one mid-priced model scored 1.183 against the best trained router's 1.057,
for free, because a cheaper point on the same curve gets a cost advantage
over a dearer one whatever exponent trades quality for cost. The exponent
derivation and the break-even analysis are kept in docs/design-decisions.md
as the record; they were correct about the corner they examined and silent
about the middle of the curve.

**Why not the other miners.** Emissions are already relative: weights are
normalised across miners by Yuma. The score itself must be absolute so that
no miner can move another's by showing up, leaving, or registering copies
(I4), so that two validators seeing different fields compute the same score
for the same proof (I1), and so that a field of constant policies is paid
nothing rather than the best of them being paid everything.

**Where the miner's own tradeoff lives.** The objective a router should
optimise — quality minus λ·cost minus μ·variance under a budget — is the
miner's training problem, and each miner picks its own λ and μ
(`TRAINING_COST_LAMBDA`). The subnet decides what it pays for, not how a
router should reason.

Variance on the miner's side is the Wilson lower bound; a noisy router earns
less. Variance on the frontier's side is the frame's warmth: a frontier built
from the prior alone shows every decent model as "headroom", so scores scale
with how well the hull is measured (`Frontier.confidence`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fugal_subnet.config import (
    BURN_IN_QUESTIONS,
    EVIDENCE_HALF_LIFE,
    SCORE_QUALITY_FLOOR,
    SLICE_SIZE,
)
from fugal_subnet.evidence import Evidence, accumulate_epoch, apply_miss
from fugal_subnet.frontier import Frontier
from fugal_subnet.head_eval import HeadScore


@dataclass
class MinerRecord:
    uid: int
    # Operator identity. Records are keyed by UID, but Bittensor recycles UIDs
    # when a miner deregisters — without this, a newly registered miner would
    # inherit the previous occupant's liveness standing and epoch history.
    hotkey: str = ""
    epochs_seen: int = 0
    epochs_missed: int = 0
    current_head_hash: str = ""
    accuracy: float = 0.0
    # Accuracy relative to the frontier at this miner's cost (1.0 = matches the
    # best constant policy at that price). Informational; the score is headroom.
    quality: float = 0.0
    # Reference cost over miner cost against the best single model. Kept for the
    # reveal because operators read it; it decides nothing.
    thrift: float = 0.0
    headroom: float = 0.0
    reference_accuracy: float = 0.0
    cost_per_question: float = 0.0
    composite_score: float = 0.0
    wilson_lcb: float = 0.0
    evidence: Evidence | None = None


@dataclass
class ScoringState:
    records: dict[int, MinerRecord] = field(default_factory=dict)
    epoch_count: int = 0


def update_scores(
    state: ScoringState,
    epoch_scores: dict[int, HeadScore],
    head_hashes: dict[int, str],
    frontier: Frontier,
    hotkeys: dict[int, str] | None = None,
    n_questions: int = 0,
    pool_size: float = 0.0,
) -> ScoringState:
    """Update scores with new epoch results using evidence accumulation.

    Args:
        frontier: This epoch's constant-policy frontier (fugal_subnet.frontier).
        hotkeys: {uid: hotkey} so a recycled UID drops the old occupant's record.
        pool_size: Distinct questions available, capping the effective sample size.
    """
    state.epoch_count += 1
    n_expected = n_questions or SLICE_SIZE
    hotkeys = hotkeys or {}

    # Drop records whose UID has been reassigned to a different operator.
    for uid, rec in list(state.records.items()):
        current = hotkeys.get(uid)
        if current and rec.hotkey and current != rec.hotkey:
            del state.records[uid]

    active_uids = set(epoch_scores.keys())
    for uid in list(state.records.keys()):
        if uid not in active_uids:
            rec = state.records[uid]
            rec.epochs_missed += 1
            if rec.evidence is not None:
                rec.evidence = apply_miss(rec.evidence, n_expected, EVIDENCE_HALF_LIFE)
                _refresh(rec, frontier)

    for uid, score in epoch_scores.items():
        if uid not in state.records:
            state.records[uid] = MinerRecord(uid=uid)

        rec = state.records[uid]
        rec.hotkey = hotkeys.get(uid, rec.hotkey)
        rec.current_head_hash = head_hashes.get(uid, "")
        rec.accuracy = score.accuracy
        rec.epochs_seen += 1
        rec.epochs_missed = 0

        rec.evidence = accumulate_epoch(
            rec.evidence,
            weights_hash=rec.current_head_hash,
            n_correct=score.n_correct,
            n_total=score.n_scored,
            cost=score.total_head_cost,
            ref_cost=score.total_oracle_cost,
            half_life=EVIDENCE_HALF_LIFE,
            pool_size=pool_size,
        )
        _refresh(rec, frontier)

    return state


def _refresh(rec: MinerRecord, frontier: Frontier) -> None:
    ev = rec.evidence
    if ev is None:
        return
    rec.wilson_lcb = ev.wilson_lcb
    rec.thrift = ev.thrift
    rec.cost_per_question = ev.cost_per_question
    rec.reference_accuracy = frontier.accuracy_at(ev.cost_per_question)
    rec.quality = quality_term(ev.wilson_lcb, rec.reference_accuracy)
    rec.headroom = ev.wilson_lcb - rec.reference_accuracy
    rec.composite_score = composite(ev, frontier)


def quality_term(accuracy_lcb: float, reference_accuracy: float) -> float:
    """Accuracy relative to the best constant policy at the miner's price.

    A reference of zero means no constant policy answers anything at that
    price — a real state of the world, not a division to paper over.
    """
    if reference_accuracy <= 1e-9:
        return 0.0
    return accuracy_lcb / reference_accuracy


def composite(ev: Evidence, frontier: Frontier) -> float:
    """Headroom above the frontier, ramped in, scaled by frontier confidence.

    Zero when the miner's lower-bound accuracy is below SCORE_QUALITY_FLOOR of
    the frontier's maximum — a router that gives up that much of the best
    model's accuracy has not delivered the product however cheap it is — and
    zero for any non-positive headroom, which is every constant policy.
    """
    lcb = ev.wilson_lcb
    if lcb <= 0.0:
        return 0.0
    if lcb < SCORE_QUALITY_FLOOR * frontier.max_accuracy:
        return 0.0
    room = lcb - frontier.accuracy_at(ev.cost_per_question)
    if room <= 0.0:
        return 0.0
    return room * burn_in_factor(ev.n_total) * frontier.confidence


def burn_in_factor(n_total: float) -> float:
    """Ramp a fresh artifact in over BURN_IN_QUESTIONS scored questions.

    Without this, evidence reset is a free penalty wash: a miner with a bad
    record flips one weight bit, the accumulator resets, and it is immediately
    back at full score. With it, recovering after a reset takes exactly as long
    as earning the position did — so reset still makes dethroning cost work,
    without also making bad records disposable.
    """
    if BURN_IN_QUESTIONS <= 0:
        return 1.0
    return min(1.0, n_total / BURN_IN_QUESTIONS)


def wilson_lower_bound(p: float, n: int, confidence: float = 0.95) -> float:
    """Wilson score interval lower bound (kept for external callers)."""
    if n == 0:
        return 0.0
    z = _z_score(confidence)
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - spread) / denominator)


def _z_score(confidence: float) -> float:
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    return table.get(confidence, 1.96)
