"""Behavioral dedup: detect and disqualify copied heads.

Runs all heads on the same hidden states, compares output distributions.
Heads producing nearly identical routing decisions are clustered; only the
earliest-committed (by block timestamp / commit hash) survives.
"""
from __future__ import annotations

import logging

import numpy as np

from fugal_subnet.config import DEDUP_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def find_duplicates(
    head_outputs: dict[int, np.ndarray],
    commit_blocks: dict[int, int],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> set[int]:
    """Find UIDs that should be disqualified as copies.

    Args:
        head_outputs: {uid: (N,) routing decision array} — the model index each
                      head chose for each question.
        commit_blocks: {uid: block_number} — when each head was first committed.
                       Lower block = earlier = gets priority.
        threshold: Cosine similarity threshold. Above this = duplicate.

    Returns:
        Set of UIDs to disqualify.
    """
    uids = sorted(head_outputs.keys())
    if len(uids) < 2:
        return set()

    n_models = max(
        (int(d.max()) + 1 for d in head_outputs.values() if len(d) > 0),
        default=1,
    )

    vectors = {}
    for uid in uids:
        decisions = head_outputs[uid]
        onehot = np.zeros((len(decisions), n_models), dtype=np.float32)
        for i, d in enumerate(decisions):
            if 0 <= d < n_models:
                onehot[i, d] = 1.0
        vectors[uid] = onehot.flatten()

    clusters: list[list[int]] = []
    clustered: set[int] = set()

    for i, uid_a in enumerate(uids):
        if uid_a in clustered:
            continue
        cluster = [uid_a]
        for uid_b in uids[i + 1:]:
            if uid_b in clustered:
                continue
            sim = _cosine_similarity(vectors[uid_a], vectors[uid_b])
            if sim >= threshold:
                cluster.append(uid_b)
                clustered.add(uid_b)
        if len(cluster) > 1:
            clusters.append(cluster)
            clustered.add(uid_a)

    disqualified: set[int] = set()
    for cluster in clusters:
        cluster.sort(key=lambda uid: commit_blocks.get(uid, float("inf")))
        survivor = cluster[0]
        copies = cluster[1:]
        logger.info(
            "Dedup cluster: survivor UID %d (block %d), copies: %s",
            survivor, commit_blocks.get(survivor, -1),
            [(uid, commit_blocks.get(uid, -1)) for uid in copies],
        )
        disqualified.update(copies)

    return disqualified


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
