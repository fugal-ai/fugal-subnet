"""Scoring system: raw epoch scoring with composite weights.

Miners are scored on current-epoch performance. Weight capping (in rewards.py)
provides stability — no EWMA smoothing needed. Wilson LCB is computed for
diagnostics but not used in weight calculation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from fugal_subnet.config import (
    WILSON_CONFIDENCE,
    COMPOSITE_W_ACC, COMPOSITE_W_COST, COMPOSITE_W_KL,
)
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
    """Update scores with new epoch results.

    Uses raw epoch scores — no smoothing. Weight capping in rewards.py
    provides the stability guarantees.

    Args:
        state: Current scoring state.
        epoch_scores: {uid: HeadScore} from this epoch's evaluation.
        head_hashes: {uid: sha256_hex} of each miner's submitted head.
        n_questions: Number of scoreable questions this epoch (Wilson LCB
            sample size). Falls back to each score's correct_mask length.

    Returns:
        Updated ScoringState.
    """
    state.epoch_count += 1

    active_uids = set(epoch_scores.keys())
    for uid in list(state.records.keys()):
        if uid not in active_uids:
            state.records[uid].epochs_missed += 1

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

        rec.composite_score = (
            COMPOSITE_W_ACC * rec.accuracy
            + COMPOSITE_W_COST * rec.cost_efficiency
            + COMPOSITE_W_KL * _normalize_kl(rec.kl_score)
        )
        n = n_questions or len(score.correct_mask)
        rec.wilson_lcb = wilson_lower_bound(rec.accuracy, n, WILSON_CONFIDENCE)

    return state


def wilson_lower_bound(p: float, n: int, confidence: float = 0.95) -> float:
    """Wilson score interval lower bound (diagnostic only).

    Provides a conservative estimate of true performance with limited samples.
    """
    if n == 0:
        return 0.0
    z = _z_score(confidence)
    denominator = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - spread) / denominator)


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels."""
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    return table.get(confidence, 1.96)


def _normalize_kl(kl: float) -> float:
    """Map KL score from (-inf, 0] to (0, ~0.73] for composite scoring.

    KL scores are negative (higher = better); kl=0 (perfect match) maps to
    1/(1+e^-1) ~= 0.731. Sigmoid keeps very bad KL from dragging the composite
    below what accuracy/cost earn. Exponent is clamped to avoid overflow on
    adversarial inputs.
    """
    return 1.0 / (1.0 + math.exp(min(700.0, -kl - 1.0)))
