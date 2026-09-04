"""Scoring: quality per dollar against the best single model.

    quality = wilson_lcb(accuracy) / acc_best
    thrift  = ref_cost / miner_cost
    score   = quality**w * thrift**(1-w) * burn_in        (w = 0.8, derived)

A score of 1.0 means "matched the best single model's quality per dollar";
above 1.0 means "beat it". That is the product claim stated directly, which is
the point — the number a miner optimises should be the number the subnet
exists to produce.

**Why a weighted geometric mean and not a weighted sum.** The old composite was
`0.55*accuracy + 0.35*cost + 0.10*kl`. Additive terms *substitute*: a miner
trades accuracy for cost at whatever exchange rate the designer picked, and
nothing about routing says one accuracy point is worth 0.64 cost points. Worse,
each degenerate strategy scores well on one axis — route everything to the
cheapest model, or everything to the best — and collects that term's weight
regardless of the other. Under a product neither axis can rescue the other:
both degenerate strategies score badly.

The exponent w is derived, not picked. "Match frontier quality at a fraction of
the cost" makes quality a near-constraint, so giving up 40% of quality must not
outscore matching the best model at its own price — which forces w > 0.778.
An unweighted sqrt fails that test (it scores such a router 1.095 against a
quality match's 1.000); w = 0.8 scores it 0.951. See SCORE_QUALITY_EXPONENT.

**Why the reference is the best single model.** It needs only per-model
marginals, so it is well-estimated within a few epochs and stable at any miner
count — see `reference_frame.py`. A per-question oracle measures pure routing
skill more precisely but is unreachable in practice, needs a dense
question-by-model matrix, and compresses every real router into a narrow band.

The KL term is gone. Under the TEE architecture the validator has no oracle
distribution to compare against, so it was hardcoded to zero — which
`_normalize_kl` mapped to a constant 0.0731 added to every miner. A constant
cannot rank; it only flattened the gradient between good and bad miners in the
proportional weight computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fugal_subnet.config import (
    BURN_IN_QUESTIONS,
    EVIDENCE_HALF_LIFE,
    SCORE_QUALITY_CAP,
    SCORE_QUALITY_EXPONENT,
    SCORE_THRIFT_CAP,
    SLICE_SIZE,
)
from fugal_subnet.evidence import Evidence, accumulate_epoch, apply_miss
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
    quality: float = 0.0
    thrift: float = 0.0
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
    acc_best: float,
    hotkeys: dict[int, str] | None = None,
    n_questions: int = 0,
    pool_size: float = 0.0,
) -> ScoringState:
    """Update scores with new epoch results using evidence accumulation.

    Args:
        acc_best: Reference accuracy — the best single model's lower bound.
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
                _refresh(rec, acc_best)

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
        _refresh(rec, acc_best)

    return state


def _refresh(rec: MinerRecord, acc_best: float) -> None:
    ev = rec.evidence
    if ev is None:
        return
    rec.wilson_lcb = ev.wilson_lcb
    rec.quality = quality_term(ev.wilson_lcb, acc_best)
    rec.thrift = ev.thrift
    rec.composite_score = composite(ev, acc_best)


def quality_term(accuracy_lcb: float, acc_best: float) -> float:
    """Accuracy relative to the reference model.

    An `acc_best` of zero means no model in the pool answers anything, so there
    is nothing to route toward and nobody has demonstrated routing skill. That
    is a real state of the world, not a division to paper over.
    """
    if acc_best <= 1e-9:
        return 0.0
    return accuracy_lcb / acc_best


def composite(ev: Evidence, acc_best: float) -> float:
    """quality^w * thrift^(1-w), ramped in over the burn-in period.

    See config.SCORE_QUALITY_EXPONENT for where w comes from — it is derived
    from the product claim rather than chosen.
    """
    w = SCORE_QUALITY_EXPONENT
    quality = min(max(quality_term(ev.wilson_lcb, acc_best), 0.0), SCORE_QUALITY_CAP)
    thrift = min(max(ev.thrift, 0.0), SCORE_THRIFT_CAP)
    if quality <= 0.0 or thrift <= 0.0:
        return 0.0
    return (quality ** w) * (thrift ** (1.0 - w)) * burn_in_factor(ev.n_total)


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
