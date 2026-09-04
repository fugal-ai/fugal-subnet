"""Scoring system: evidence-accumulated composite with Wilson LCB.

Miners accumulate evidence across epochs, keyed by head artifact.
Changing the head resets the accumulator. Missing an epoch counts as
n_expected tasks scored 0. The composite score uses Wilson LCB for
accuracy, with cost efficiency and KL from accumulated evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fugal_subnet.config import (
    COMPOSITE_W_ACC,
    COMPOSITE_W_COST,
    COMPOSITE_W_KL,
    EVIDENCE_HALF_LIFE,
    SLICE_SIZE,
)
from fugal_subnet.evidence import Evidence, accumulate_epoch, apply_miss
from fugal_subnet.head_eval import HeadScore


@dataclass
class MinerRecord:
    uid: int
    epochs_seen: int = 0
    epochs_missed: int = 0
    current_head_hash: str = ""
    accuracy: float = 0.0
    cost_efficiency: float = 0.0
    kl_score: float = 0.0
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
    n_questions: int = 0,
) -> ScoringState:
    """Update scores with new epoch results using evidence accumulation."""
    state.epoch_count += 1
    n_expected = n_questions or SLICE_SIZE

    active_uids = set(epoch_scores.keys())
    for uid in list(state.records.keys()):
        if uid not in active_uids:
            rec = state.records[uid]
            rec.epochs_missed += 1
            if rec.evidence is not None:
                rec.evidence = apply_miss(rec.evidence, n_expected, EVIDENCE_HALF_LIFE)
                rec.wilson_lcb = rec.evidence.wilson_lcb
                rec.composite_score = _composite_from_evidence(rec.evidence)

    for uid, score in epoch_scores.items():
        if uid not in state.records:
            state.records[uid] = MinerRecord(uid=uid)

        rec = state.records[uid]
        rec.current_head_hash = head_hashes.get(uid, "")
        rec.accuracy = score.accuracy
        rec.cost_efficiency = score.cost_efficiency
        rec.kl_score = score.kl_score
        rec.epochs_seen += 1
        rec.epochs_missed = 0

        rec.evidence = accumulate_epoch(
            rec.evidence,
            weights_hash=rec.current_head_hash,
            n_correct=score.n_correct,
            n_total=score.n_scored,
            head_cost=score.total_head_cost,
            oracle_cost=score.total_oracle_cost,
            kl_total=score.total_kl,
            n_kl=score.n_scored,
            coverage=score.coverage,
            half_life=EVIDENCE_HALF_LIFE,
        )

        rec.wilson_lcb = rec.evidence.wilson_lcb
        rec.composite_score = _composite_from_evidence(rec.evidence)

    return state


def _composite_from_evidence(ev: Evidence) -> float:
    raw = (
        COMPOSITE_W_ACC * ev.wilson_lcb
        + COMPOSITE_W_COST * ev.cost_efficiency
        + COMPOSITE_W_KL * _normalize_kl(ev.avg_kl)
    )
    return raw * ev.avg_coverage


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


def _normalize_kl(kl: float) -> float:
    """Map KL score from (-inf, 0] to (0, ~0.73] for composite scoring."""
    if math.isnan(kl):
        return 0.0
    return 1.0 / (1.0 + math.exp(min(700.0, -kl - 1.0)))
