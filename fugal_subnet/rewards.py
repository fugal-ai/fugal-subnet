"""Reward computation: single-ranking weight vector from composite scores.

Miners compete in one pool. Weight is proportional to composite score.
Unassigned weight is burned to the subnet owner's UID, which the caller
resolves from chain (`subtensor.get_subnet_owner_hotkey` at the epoch
boundary). It used to be a hard-coded 0, which is the owner's UID only by
convention and only until the owner's hotkey is ever re-registered
(diligence 2026-10, finding A5).
"""
from __future__ import annotations

import logging

import numpy as np

from fugal_subnet.config import LIVENESS_MAX_MISSED_EVIDENCE, MAX_WEIGHT_DELTA
from fugal_subnet.scoring import MinerRecord

logger = logging.getLogger(__name__)


def compute_weights(
    records: dict[int, MinerRecord],
    dedup_disqualified: set[int] | None = None,
    paid_fraction: float = 1.0,
    *,
    burn_uid: int,
) -> tuple[list[int], list[float]]:
    """Compute the weight vector for set_weights().

    Args:
        records: {uid: MinerRecord} with accumulated scores.
        dedup_disqualified: UIDs disqualified by behavioral dedup.
        paid_fraction: share of the emission the miners split this epoch; the
            rest burns. The validator passes the frontier's confidence here.
            A factor common to every miner's score cancels when the scores are
            normalised, so the confidence discount in `composite` could never
            reach the weight vector on its own — the first live frontier epoch
            paid a 0.08-confidence frontier out in full (measured, testnet
            552, e00022118). The burn has to happen here, after normalising.
        burn_uid: where unassigned weight goes — the subnet owner's UID,
            resolved from chain by the caller. Keyword-only and required so
            no caller inherits a silent 0.

    Returns:
        (uids, weights) — parallel lists for subtensor.set_weights().
        Weights sum to ~1.0. Unassigned weight goes to `burn_uid`.
    """
    if dedup_disqualified is None:
        dedup_disqualified = set()
    paid_fraction = min(1.0, max(0.0, float(paid_fraction)))
    burn_uid = int(burn_uid)

    miners: list[tuple[int, float]] = []
    for uid, rec in records.items():
        if uid in dedup_disqualified:
            continue
        if uid == burn_uid:
            continue
        if rec.epochs_missed >= LIVENESS_MAX_MISSED_EVIDENCE:
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
                weights[uid] = paid_fraction * s / total

    assigned = sum(weights.values())
    burn = 1.0 - assigned
    if burn > 1e-12:
        weights[burn_uid] = weights.get(burn_uid, 0.0) + burn

    if not weights:
        weights[burn_uid] = 1.0

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
