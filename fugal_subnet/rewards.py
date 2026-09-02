"""Reward computation: single-ranking weight vector from composite scores.

Miners compete in one pool. Weight is proportional to composite score.
Unassigned weight is burned to UID 0 (Greevils pattern).
"""
from __future__ import annotations

import logging

import numpy as np

from fugal_subnet.config import LIVENESS_MAX_MISSED, MAX_WEIGHT_DELTA
from fugal_subnet.scoring import MinerRecord

logger = logging.getLogger(__name__)

BURN_UID = 0


def compute_weights(
    records: dict[int, MinerRecord],
    dedup_disqualified: set[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Compute the weight vector for set_weights().

    Args:
        records: {uid: MinerRecord} with accumulated scores.
        dedup_disqualified: UIDs disqualified by behavioral dedup.

    Returns:
        (uids, weights) — parallel lists for subtensor.set_weights().
        Weights sum to ~1.0. Unassigned weight goes to BURN_UID.
    """
    if dedup_disqualified is None:
        dedup_disqualified = set()

    miners: list[tuple[int, float]] = []
    for uid, rec in records.items():
        if uid in dedup_disqualified:
            continue
        if uid == BURN_UID:
            continue
        if rec.epochs_missed >= LIVENESS_MAX_MISSED:
            continue
        miners.append((uid, rec.composite_score))

    weights: dict[int, float] = {}
    if miners:
        miners.sort(key=lambda x: x[1], reverse=True)
        scores = np.array([s for _, s in miners])
        scores = np.clip(scores, 0, None)
        total = scores.sum()
        if total > 0:
            for (uid, _), s in zip(miners, scores):
                weights[uid] = s / total

    assigned = sum(weights.values())
    burn = 1.0 - assigned
    if burn > 1e-12:
        weights[BURN_UID] = weights.get(BURN_UID, 0.0) + burn

    if not weights:
        weights[BURN_UID] = 1.0

    uids = sorted(weights.keys())
    return uids, [weights[uid] for uid in uids]


def cap_weight_change(
    new_uids: list[int],
    new_weights: list[float],
    prev_uids: list[int],
    prev_weights: list[float],
    max_delta: float = MAX_WEIGHT_DELTA,
) -> tuple[list[int], list[float]]:
    """Clamp per-UID weight change to prevent single-epoch manipulation.

    The final renormalization (so weights sum to 1) can stretch a clamped
    delta slightly past max_delta; the cap is a stability damper, not an
    exact invariant.
    """
    prev_map = dict(zip(prev_uids, prev_weights))
    all_uids = sorted(set(new_uids) | set(prev_uids))
    new_map = dict(zip(new_uids, new_weights))

    capped: dict[int, float] = {}
    for uid in all_uids:
        old_w = prev_map.get(uid, 0.0)
        target_w = new_map.get(uid, 0.0)
        delta = target_w - old_w
        clamped = old_w + max(min(delta, max_delta), -max_delta)
        capped[uid] = max(0.0, clamped)

    total = sum(capped.values())
    if total > 0:
        for uid in capped:
            capped[uid] /= total

    uids = sorted(capped.keys())
    return uids, [capped[uid] for uid in uids]
