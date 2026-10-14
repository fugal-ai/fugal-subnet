"""Dedup: detect and disqualify copied heads.

Three clauses, any one of which marks a head a copy of a senior one:

1. **Behavioural.** Routing decisions, taken from the attested proof (the
   model each head chose for each scored question, in a global model index),
   agree on at least `DEDUP_SIMILARITY_THRESHOLD` of the slice.
2. **Parametric.** The heads' weights sit within `DEDUP_PARAM_DISTANCE`
   relative Frobenius distance of each other (`head_distance.py`). A copier
   can nudge decisions past clause 1 by hand; it cannot move the weights far
   without becoming a different router (diligence 2026-10, finding 5).
3. **Epsilon.** `epsilon_disqualified`: a head agreeing with a senior one on
   at least `DEDUP_EPSILON_AGREEMENT` of decisions must beat its composite
   score by `DEDUP_EPSILON` or it is disqualified. Near-copies have to be
   improvements.

In every clause the senior head (lowest `commitments.seniority_key`) survives.
"""
from __future__ import annotations

import logging

import numpy as np

from fugal_subnet.config import (
    DEDUP_EPSILON,
    DEDUP_EPSILON_AGREEMENT,
    DEDUP_PARAM_DISTANCE,
    DEDUP_SIMILARITY_THRESHOLD,
)
from fugal_subnet.head_distance import relative_distance

logger = logging.getLogger(__name__)


def find_duplicates(
    head_outputs: dict[int, np.ndarray],
    commit_blocks: dict[int, int],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    head_params: dict[int, object] | None = None,
    param_threshold: float = DEDUP_PARAM_DISTANCE,
) -> set[int]:
    """Find UIDs that should be disqualified as copies.

    Args:
        head_outputs: {uid: (N,) routing decision array} — the model index each
                      head chose for each question.
        commit_blocks: {uid: seniority} — lower is senior and survives
                       (commitments.seniority_key; a missing uid is junior to all).
        threshold: Cosine similarity of one-hot decisions at or above which two
                   heads are one head.
        head_params: {uid: HeadArtifact or {"W","b","models"}}; when both heads
                     of a pair are present, a relative parameter distance below
                     `param_threshold` also makes them one head.

    Returns:
        Set of UIDs to disqualify.
    """
    uids = sorted(head_outputs.keys())
    if len(uids) < 2:
        return set()

    # Decision arrays must already be indexed in a GLOBAL model space (see
    # neurons/validator._extract_routing_decisions). When each miner numbered
    # its own models, index 0 meant a different model for every miner, and the
    # comparison broke in both directions: two miners each routing 100% to a
    # different single model produced identical all-zero vectors and were
    # clustered as clones, while a real copy that re-routed one question to an
    # alphabetically-earlier model renumbered its whole vector and escaped —
    # 96.7% identical routing, zero similarity detected.
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
            same = sim >= threshold
            if not same and head_params and uid_a in head_params and uid_b in head_params:
                same = relative_distance(head_params[uid_a], head_params[uid_b]) < param_threshold
            if same:
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


def epsilon_disqualified(
    head_outputs: dict[int, np.ndarray],
    commit_blocks: dict[int, int],
    scores: dict[int, float],
    agreement: float = DEDUP_EPSILON_AGREEMENT,
    epsilon: float = DEDUP_EPSILON,
) -> set[int]:
    """UIDs whose head is a near-copy of a senior head and does not beat it.

    For every pair whose one-hot routing vectors agree at or above
    `agreement`, the junior head (higher seniority key) must have a composite
    score above the senior's by more than `epsilon` relative, or it is
    disqualified. Subnet 9 used this rule, at about 0.5%, to stop
    copy-and-tweak: a copy that reroutes a few questions is a different head
    only if it is a BETTER head. Deterministic: pairs are visited in sorted
    uid order and the decision for each depends only on its two members.
    """
    uids = sorted(set(head_outputs) & set(scores))
    if len(uids) < 2:
        return set()
    n_models = max((int(d.max()) + 1 for d in head_outputs.values() if len(d) > 0), default=1)
    vectors = {}
    for uid in uids:
        decisions = head_outputs[uid]
        onehot = np.zeros((len(decisions), n_models), dtype=np.float32)
        for i, d in enumerate(decisions):
            if 0 <= d < n_models:
                onehot[i, d] = 1.0
        vectors[uid] = onehot.flatten()

    def seniority(uid):
        return commit_blocks.get(uid, float("inf"))

    out: set[int] = set()
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            if _cosine_similarity(vectors[a], vectors[b]) < agreement:
                continue
            senior, junior = (a, b) if seniority(a) <= seniority(b) else (b, a)
            if scores[junior] <= scores[senior] * (1.0 + epsilon):
                logger.info(
                    "Epsilon rule: UID %d routes like senior UID %d and does not "
                    "beat it by %.2f%% (%.5f vs %.5f)",
                    junior, senior, 100 * epsilon, scores[junior], scores[senior],
                )
                out.add(junior)
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
